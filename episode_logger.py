import numpy as np
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font


class EpisodeLogger:
    def __init__(self, path="training_log.xlsx", theta_path=None):
        self.path = path

        # Maak standaard automatisch een bijpassende .npy-bestandsnaam.
        # Bijvoorbeeld: training_log.xlsx -> training_log_theta_final.npy
        if theta_path is None:
            excel_path = Path(path)
            theta_path = excel_path.with_name(
                f"{excel_path.stem}_theta_final.npy"
            )

        self.theta_path = str(theta_path)
        self.rows = []
        self.theta_rows = []

    def log_episode(self, episode, won, points, theta=None):
        self.rows.append((episode, int(won), points))

        if theta is not None:
            flat = np.asarray(theta, dtype=float).ravel().tolist()
            self.theta_rows.append((episode, flat))

    def stop(self, theta=None):
        """Schrijf Excel-log én sla de uiteindelijke theta op als .npy-bestand."""
        bold = Font(bold=True)
        wb = Workbook()

        # Tabblad 1: episodes
        ws = wb.active
        ws.title = "Episodes"
        ws.append(["episode", "won", "points"])

        for row in self.rows:
            ws.append(row)

        for cell in ws[1]:
            cell.font = bold

        # Tabblad 2: theta na elke episode
        if self.theta_rows:
            n = len(self.theta_rows[0][1])

            if n + 1 > 16384:
                raise ValueError(
                    "Theta is te groot voor één Excel-rij (max 16383 waarden)."
                )

            ws_t = wb.create_sheet("Theta")
            ws_t.append(["episode"] + [f"theta_{k}" for k in range(n)])

            for episode, flat_theta in self.theta_rows:
                ws_t.append([episode] + flat_theta)

            for cell in ws_t[1]:
                cell.font = bold

        # Tabblad 3: uiteindelijke theta in originele vorm
        if theta is not None:
            theta_array = np.asarray(theta)

            ws_f = wb.create_sheet("Theta_final")
            for row in np.atleast_2d(theta_array).tolist():
                ws_f.append(row)

            # Apart bestand dat je direct weer met np.load kunt inladen
            np.save(self.theta_path, theta_array)

        # Tabblad 4: samenvatting
        ws_s = wb.create_sheet("Samenvatting")

        last = len(self.rows) + 1
        last_100_start = max(2, last - 99)

        ws_s.append(["Aantal episodes", f"=COUNT(Episodes!A2:A{last})"])
        ws_s.append(["Aantal gewonnen", f"=COUNTIF(Episodes!B2:B{last}, 1)"])
        ws_s.append(["Winstpercentage", "=IFERROR(B2/B1,0)"])
        ws_s.append([
            "Gemiddelde punten",
            f"=IFERROR(AVERAGE(Episodes!C2:C{last}),0)"
        ])
        ws_s.append([
                    "Winrate laatste 100",
                    f"=IFERROR(COUNTIF(Episodes!B{last_100_start}:B{last}),1)"
                ])
        
        ws_s.append([
            "Winrate laatste 100",
            f"=IFERROR(COUNTIF(Episodes!B{last_100_start}:B{last}),1) / 100"
        ])

        ws_s["B3"].number_format = "0.0%"
        ws_s["B4"].number_format = "0.00"
        ws_s["B5"].number_format = "0.0%"

        for row in range(1, 6):
            ws_s.cell(row=row, column=1).font = bold

        ws_s.column_dimensions["A"].width = 24

        wb.save(self.path)