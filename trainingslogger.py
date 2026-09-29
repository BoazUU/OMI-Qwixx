from pathlib import Path
from datetime import datetime
import csv
import json
import numpy as np


class TrainingLogger:
    def __init__(self, save_directory="training_results", metadata=None):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)

        self.metadata = metadata or {}
        self.metadata["created_at"] = datetime.now().isoformat()

        self.episodes = []
        self.steps = []
        self.theta_history = []

        self.current_episode = None
        self.current_episode_steps = 0

    def start_episode(self, episode, epsilon=None):
        """Start een nieuwe episode."""
        self.current_episode = {
            "episode": episode,
            "epsilon_start": epsilon,
            "epsilon_end": epsilon,
            "total_reward": 0.0,
            "steps": 0,
            "won": None,
            "status": None,
            "finished": False,
            "loss_sum": 0.0,
            "loss_mean": None,
        }

        self.current_episode_steps = 0

    def log_step(
        self,
        state=None,
        action=None,
        reward=0.0,
        q_value=None,
        target=None,
        delta=None,
        epsilon=None,
        theta=None,
        finished=False,
        status=None,
    ):
        """Sla informatie van één trainingsstap op."""

        if self.current_episode is None:
            raise RuntimeError(
                "Start eerst een episode met start_episode()."
            )

        loss = None
        if delta is not None:
            loss = float(delta) ** 2

        step_number = self.current_episode_steps

        step_data = {
            "episode": self.current_episode["episode"],
            "step": step_number,
            "action": repr(action),
            "reward": float(reward),
            "q_value": None if q_value is None else float(q_value),
            "target": None if target is None else float(target),
            "delta": None if delta is None else float(delta),
            "loss": loss,
            "epsilon": None if epsilon is None else float(epsilon),
            "finished": bool(finished),
            "status": status,
        }

        self.steps.append(step_data)

        self.current_episode_steps += 1
        self.current_episode["steps"] += 1
        self.current_episode["total_reward"] += float(reward)
        self.current_episode["epsilon_end"] = epsilon
        self.current_episode["status"] = status
        self.current_episode["finished"] = bool(finished)

        if loss is not None:
            self.current_episode["loss_sum"] += loss

        if theta is not None:
            self.theta_history.append({
                "episode": self.current_episode["episode"],
                "step": step_number,
                "theta": np.asarray(theta, dtype=float).copy(),
            })

    def end_episode(self, won=None, status=None, theta=None):
        """Sluit de huidige episode af."""

        if self.current_episode is None:
            raise RuntimeError(
                "Er is geen actieve episode."
            )

        if status is not None:
            self.current_episode["status"] = status

        self.current_episode["won"] = (
            None if won is None else bool(won)
        )

        if self.current_episode["steps"] > 0:
            self.current_episode["loss_mean"] = (
                self.current_episode["loss_sum"]
                / self.current_episode["steps"]
            )

        self.episodes.append(self.current_episode.copy())

        if theta is not None:
            self.theta_history.append({
                "episode": self.current_episode["episode"],
                "step": self.current_episode["steps"],
                "theta": np.asarray(theta, dtype=float).copy(),
            })

        self.current_episode = None
        self.current_episode_steps = 0

    def save(self):
        """Sla alle verzamelde trainingsinformatie op."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = self.save_directory / f"training_{timestamp}.json"
        episodes_csv_path = self.save_directory / f"episodes_{timestamp}.csv"
        steps_csv_path = self.save_directory / f"steps_{timestamp}.csv"
        theta_path = self.save_directory / f"theta_{timestamp}.npz"

        # JSON-bestand
        json_data = {
            "metadata": self.metadata,
            "episodes": self.episodes,
            "steps": self.steps,
            "summary": self.summary(),
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(json_data, file, indent=4, default=str)

        # Episodes als CSV
        self._save_csv(
            episodes_csv_path,
            self.episodes
        )

        # Stappen als CSV
        self._save_csv(
            steps_csv_path,
            self.steps
        )

        # Theta-waarden als NumPy-bestand
        if self.theta_history:
            theta_values = np.array([
                item["theta"]
                for item in self.theta_history
            ])

            theta_metadata = np.array([
                [
                    item["episode"],
                    item["step"]
                ]
                for item in self.theta_history
            ])

            np.savez_compressed(
                theta_path,
                theta=theta_values,
                metadata=theta_metadata
            )

        return {
            "json": str(json_path),
            "episodes_csv": str(episodes_csv_path),
            "steps_csv": str(steps_csv_path),
            "theta": str(theta_path) if self.theta_history else None,
        }

    @staticmethod
    def _save_csv(path, rows):
        """Sla een lijst met dictionaries op als CSV."""
        if not rows:
            return

        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(rows)

    def summary(self):
        """Geef een samenvatting van de training."""

        if not self.episodes:
            return {
                "episodes": 0,
                "wins": 0,
                "winrate": None,
                "average_reward": None,
                "average_steps": None,
                "average_loss": None,
            }

        rewards = [
            episode["total_reward"]
            for episode in self.episodes
        ]

        steps = [
            episode["steps"]
            for episode in self.episodes
        ]

        losses = [
            episode["loss_mean"]
            for episode in self.episodes
            if episode["loss_mean"] is not None
        ]

        wins = [
            episode
            for episode in self.episodes
            if episode["won"] is True
        ]

        return {
            "episodes": len(self.episodes),
            "wins": len(wins),
            "winrate": len(wins) / len(self.episodes),
            "average_reward": float(np.mean(rewards)),
            "average_steps": float(np.mean(steps)),
            "average_loss": (
                None
                if not losses
                else float(np.mean(losses))
            ),
        }