import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")

import analisis_disdro as analysis
import utilidades_disdrometro as disd
from crear_varsDisdroBD_names_corrected import build_corrected_rows


def make_disdro_csv(path: Path, times: pd.DatetimeIndex) -> None:
    data = {
        "fecha_hora": times.strftime("%d.%m.%y;%H:%M:%S"),
        "cliente": [417] * len(times),
        disd.PRECIPITATION_COLUMN: [1.0] * len(times),
        disd.QUALITY_COLUMN: [99.0] * len(times),
        "var16": 99 + np.arange(1, len(times) + 1, dtype=np.int32),
    }
    for column in disd.PRECIPITATION_SPECTRUM_COLUMNS:
        data[column] = np.zeros(len(times), dtype=np.int32)
    data["var42"] = np.full(len(times), 99, dtype=np.int32)
    data["var43"] = np.arange(1, len(times) + 1, dtype=np.int32)
    pd.DataFrame(data).to_csv(path, index=False, encoding="latin-1")


def make_index_file(
    path: Path, variable: str, level: float, times: pd.DatetimeIndex
) -> None:
    lat = np.array([8.0, 6.0, 4.0, 2.0])
    lon = np.array([283.0, 284.0, 285.0, 286.0])
    base = np.arange(len(times), dtype=float)[:, None, None, None]
    spatial = lat[None, None, :, None] * 0.01 + lon[None, None, None, :] * 0.001
    values = base + spatial
    ds = xr.Dataset(
        {variable: (("time", "level", "lat", "lon"), values)},
        coords={"time": times, "level": [level], "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


class ReproducibilityTests(unittest.TestCase):
    def test_metadata_and_counts_are_direct_classes(self):
        meta = analysis.build_default_spectrum_metadata()
        self.assertEqual(len(meta), 440)
        series = pd.Series(0.0, index=meta["var"])
        series.loc["var42"] = 99
        series.loc["var43"] = 5
        counts = analysis.build_counts_dataframe(series, meta)
        self.assertEqual(counts["cuentas"].sum(), 104)
        self.assertEqual(set(counts["cuentas"]), {5.0, 99.0})

    def test_quality_99_and_spectral_99_are_not_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            times = pd.date_range("2022-01-01", periods=2, freq="MS")
            make_disdro_csv(results / "Disdrodata_417_2022.csv", times)
            chunks = list(
                disd.iter_disdro_chunks(
                    resultados_dir=results,
                    precipitation_only=True,
                    chunksize=1,
                    usecols=["fecha_hora", "var8", "var42"],
                )
            )
            self.assertEqual(sum(len(chunk) for _, chunk in chunks), 2)
            self.assertTrue(all(chunk["var42"].iloc[0] == 99 for _, chunk in chunks))

    def test_incremental_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            times = pd.date_range("2022-01-01", periods=2, freq="MS")
            make_disdro_csv(results / "Disdrodata_417_2022.csv", times)
            meta = analysis.build_default_spectrum_metadata()
            totals, monthly, periods, quality = analysis.aggregate_filtered_counts(
                results, meta, chunk_size=1
            )
            self.assertEqual(totals.loc["var42"], 198)
            self.assertEqual(monthly[1].loc["var42"], 99)
            self.assertEqual(len(periods), 2)
            self.assertEqual(quality["filtered_minutes"], 2)

    def test_negative_particle_counts_are_reported_and_zeroed(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            path = results / "Disdrodata_417_2022.csv"
            times = pd.date_range("2022-01-01", periods=2, freq="MS")
            make_disdro_csv(path, times)
            data = pd.read_csv(path, encoding="latin-1")
            data.loc[0, "var43"] = -10
            data.to_csv(path, index=False, encoding="latin-1")

            totals, _, _, quality = analysis.aggregate_filtered_counts(
                results, analysis.build_default_spectrum_metadata(), chunk_size=1
            )
            self.assertEqual(totals.loc["var43"], 2)
            self.assertEqual(quality["negative_spectrum_cells_removed"], 1)
            self.assertEqual(quality["negative_spectrum_magnitude_removed"], 10)

    def test_single_noaa_coordinate_gets_finite_cell_edges(self):
        edges = analysis.coordinate_cell_edges([-75.0], (-76.9964, -73.3244))
        np.testing.assert_allclose(edges, [-76.9964, -73.3244])
        multi = analysis.coordinate_cell_edges(
            [2.5, 5.0, 7.5], (2.3966, 8.0147)
        )
        np.testing.assert_allclose(multi, [2.3966, 3.75, 6.25, 8.0147])

    def test_metadata_generator_works_without_source_file(self):
        rows = build_corrected_rows(Path("archivo_que_no_existe.csv"))
        self.assertEqual(len(rows), 440)
        self.assertEqual(rows[0][1], "var42")
        self.assertEqual(rows[-1][1], "var481")

    def test_complete_synthetic_workflow_creates_all_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "resultados"
            results.mkdir()
            noaa = root / "datos_fuente" / "noaa"
            noaa.mkdir(parents=True)
            times = pd.date_range("2022-01-01", periods=4, freq="MS")
            make_disdro_csv(results / "Disdrodata_417_2022.csv", times)
            make_index_file(noaa / "omega.mon.mean.nc", "omega", 500.0, times)
            make_index_file(noaa / "chi.mon.mean.nc", "chi", 0.2101, times)

            original_base = analysis.BASE_DIR
            original_output = analysis.OUTPUT_DIR
            try:
                analysis.BASE_DIR = root
                analysis.OUTPUT_DIR = root / "figuras"
                analysis.SOURCE_DATA_DIR = root / "datos_fuente"
                analysis.main()
            finally:
                analysis.BASE_DIR = original_base
                analysis.OUTPUT_DIR = original_output
                analysis.SOURCE_DATA_DIR = original_base / "datos_fuente"

            figures = sorted((root / "figuras").glob("*.png"))
            self.assertEqual(len(figures), 14)
            self.assertTrue(all(path.stat().st_size > 0 for path in figures))
            self.assertTrue((root / "figuras" / "resumen_resultados.json").exists())


if __name__ == "__main__":
    unittest.main()
