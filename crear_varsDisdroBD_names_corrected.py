import csv
from pathlib import Path

speeds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6,
          3.0, 3.4, 4.2, 5.0, 5.8, 6.6, 7.4, 8.2, 9.0, 10.0]
diameters = [0.125, 0.250, 0.375, 0.500, 0.750, 1.000, 1.250, 1.500, 1.750,
             2.000, 2.500, 3.000, 3.500, 4.000, 4.500, 5.000, 5.500, 6.000,
             6.500, 7.000, 7.500, 8.000]

def fmt_speed(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.1f}"

def fmt_diameter(value: float) -> str:
    return f"{value:.3f}"

BASE_DIR = Path(__file__).resolve().parent


def build_corrected_rows(src_path: Path):
    """Preserve variables 1-41 when available and rebuild classes 42-481."""
    prefix = []
    if src_path.exists():
        with src_path.open("r", newline="", encoding="latin-1") as src:
            rows = list(csv.reader(src))
        for row in rows:
            if len(row) >= 2 and row[1] == "var42":
                break
            prefix.append(row)
        else:
            raise ValueError("No se encontró var42 en el archivo de origen.")

    next_index = int(prefix[-1][0]) if prefix else 41
    next_var = 42
    new_rows = []
    for diameter in diameters:
        diameter_label = fmt_diameter(diameter)
        for speed in speeds:
            next_index += 1
            description = (
                f"Number of particles speed > {fmt_speed(speed)} "
                f"and diameter > {diameter_label}"
            )
            new_rows.append([str(next_index), f"var{next_var}", description, "", ""])
            next_var += 1
    return prefix + new_rows


def main() -> None:
    src_path = BASE_DIR / "varsDisdroBD_names.csv"
    dst_path = BASE_DIR / "varsDisdroBD_names_corrected.csv"
    rows = build_corrected_rows(src_path)
    with dst_path.open("w", newline="", encoding="latin-1") as dst:
        csv.writer(dst).writerows(rows)
    print(f"Archivo generado: {dst_path} ({len(rows)} filas)")


if __name__ == "__main__":
    main()
