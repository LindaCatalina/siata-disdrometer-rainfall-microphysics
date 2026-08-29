#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Linda Catalina Correa Lozano y Juan Camilo Bedoya Carmona"""

from __future__ import annotations

import math
import re
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl

# El flujo solo guarda archivos; usar un backend no interactivo evita depender
# de Tk/Tcl, de una pantalla o de la configuración gráfica del equipo.
mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm, LinearSegmentedColormap, TwoSlopeNorm, Normalize
from scipy.ndimage import zoom

try:
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - dependencia opcional
    cartopy = None
    ccrs = None
    cfeature = None

import utilidades_disdrometro as disd  # noqa: E402  (módulo del usuario)


# ---------------------------------------------------------------------
# Configuración general de estilo
# ---------------------------------------------------------------------
mpl.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "axes.titlecolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#444444",
        "text.color": "white",
        "legend.edgecolor": "none",
        "font.size": 14,
        "font.family": "DejaVu Sans",
    }
)

TITLE_SIZE = 20
LABEL_SIZE = 16
LEGEND_SIZE = 14
STATS_SIZE = 14
LINE_WIDTH = 2.2
TITLE_PAD = 48

COLOR_DIAM = "#33a1ff"  
COLOR_VEL = "#ff66cc"  
PERCENTILE_COLORS = {
    "p10": "#3b87eb",
    "p25": "#4be7ff",
    "p50": "#63ea77",
    "p75": "#ffc857",
    "p90": "#f8579a",
}
SCATTER_ALPHA = 0.85
DIV_CMAP = LinearSegmentedColormap.from_list(
    "black_blue_red",
    ["#0072ff", "#000000", "#ff3b3b"],
)
GLOBAL_CMAP = LinearSegmentedColormap.from_list(
    "black_blue_red_seq",
    ["#000000", "#0072ff", "#ff3b3b"],
)
DISDROMETER_LAT = 6.1935
DISDROMETER_LON = -75.5276
DISDROMETER_STATION_ID = "417"
DISDROMETER_STATION_NAME = "Santa Elena Radar - Disdrometro"
REGION_LAT_BOUNDS = (2.3966, 8.0147)
REGION_LON_BOUNDS = (-76.9964, -73.3244)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "figuras"
SOURCE_DATA_DIR = BASE_DIR / "datos_fuente"

if cartopy is not None:  # Mantiene los mapas auxiliares dentro del proyecto.
    cartopy.config["data_dir"] = str(SOURCE_DATA_DIR / "cartopy")

# ---------------------------------------------------------------------
# Metadatos de clases de diámetro y velocidad
# ---------------------------------------------------------------------
DIAMETER_CLASSES = [
    (0.125, 0.125),
    (0.250, 0.125),
    (0.375, 0.125),
    (0.500, 0.250),
    (0.750, 0.250),
    (1.000, 0.250),
    (1.250, 0.250),
    (1.500, 0.250),
    (1.750, 0.250),
    (2.000, 0.500),
    (2.500, 0.500),
    (3.000, 0.500),
    (3.500, 0.500),
    (4.000, 0.500),
    (4.500, 0.500),
    (5.000, 0.500),
    (5.500, 0.500),
    (6.000, 0.500),
    (6.500, 0.500),
    (7.000, 0.500),
    (7.500, 0.500),
    (8.000, np.nan),  # última clase abierta
]

SPEED_CLASSES = [
    (0.0, 0.2),
    (0.2, 0.2),
    (0.4, 0.2),
    (0.6, 0.2),
    (0.8, 0.2),
    (1.0, 0.4),
    (1.4, 0.4),
    (1.8, 0.4),
    (2.2, 0.4),
    (2.6, 0.4),
    (3.0, 0.8),
    (3.4, 0.8),
    (4.2, 0.8),
    (5.0, 0.8),
    (5.8, 0.8),
    (6.6, 0.8),
    (7.4, 0.8),
    (8.2, 0.8),
    (9.0, 1.0),
    (10.0, 10.0),
]


def center_from_edge(edge: float, width: float) -> float:
    if math.isnan(width):
        return edge + 0.5  # aproximación para la última clase abierta
    return edge + width / 2.0


DIAMETER_CENTER = {thr: center_from_edge(thr, width) for thr, width in DIAMETER_CLASSES}
SPEED_CENTER = {thr: center_from_edge(thr, width) for thr, width in SPEED_CLASSES}


# ---------------------------------------------------------------------
# Helpers de estadística ponderada
# ---------------------------------------------------------------------
def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.nan
    order = np.argsort(values[valid])
    x_sorted = values[valid][order]
    w_sorted = weights[valid][order]
    cumulative = np.cumsum(w_sorted)
    cutoff = 0.5 * w_sorted.sum()
    idx = np.searchsorted(cumulative, cutoff)
    idx = np.clip(idx, 0, len(x_sorted) - 1)
    return float(x_sorted[idx])


def weighted_percentiles(
    values: np.ndarray, weights: np.ndarray, percentiles: Sequence[float]
) -> np.ndarray:
    percentiles_array = np.asarray(percentiles, dtype=float)
    if np.any((percentiles_array < 0) | (percentiles_array > 100)):
        raise ValueError("Los percentiles deben estar entre 0 y 100")
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return np.full(len(percentiles), np.nan)
    order = np.argsort(values[valid])
    x_sorted = values[valid][order]
    w_sorted = weights[valid][order]
    cdf = np.cumsum(w_sorted)
    total = w_sorted.sum()
    if total == 0:
        return np.full(len(percentiles), np.nan)
    targets = total * (percentiles_array / 100.0)
    return np.interp(targets, cdf, x_sorted)


# ---------------------------------------------------------------------
# Carga y normalización de metadatos del espectro (vars42-481)
# ---------------------------------------------------------------------
def build_default_spectrum_metadata() -> pd.DataFrame:
    """Build the canonical 22 x 20 Thies LPM spectral-class metadata."""
    rows = []
    var_number = disd.PRECIPITATION_SPECTRUM_START
    for diameter_min, _ in DIAMETER_CLASSES:
        for speed_min, _ in SPEED_CLASSES:
            rows.append(
                {
                    "var": f"var{var_number}",
                    "speed_min": speed_min,
                    "speed_center": SPEED_CENTER[speed_min],
                    "diameter_min": diameter_min,
                    "diameter_center": DIAMETER_CENTER[diameter_min],
                }
            )
            var_number += 1
    return pd.DataFrame(rows)


def load_spectrum_metadata(csv_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(
        csv_path,
        header=None,
        names=["idx", "var", "desc", "extra1", "extra2"],
        usecols=[0, 1, 2],
        encoding="latin-1",
    )
    meta = meta.dropna(subset=["var", "desc"])
    meta = meta[meta["var"].str.match(r"var\d+")]
    speed_pattern = re.compile(r"speed\s*>\s*([0-9.]+)")
    diam_pattern = re.compile(r"diameter\s*>\s*([0-9.]+)")

    meta["speed_min"] = (
        meta["desc"].str.extract(speed_pattern)[0].str.replace(",", ".").astype(float)
    )
    meta["diameter_min"] = (
        meta["desc"].str.extract(diam_pattern)[0].str.replace(",", ".").astype(float)
    )
    meta["speed_center"] = meta["speed_min"].map(SPEED_CENTER)
    meta["diameter_center"] = meta["diameter_min"].map(DIAMETER_CENTER)
    meta = meta.dropna(subset=["speed_center", "diameter_center"])
    result = meta[["var", "speed_min", "speed_center", "diameter_min", "diameter_center"]]
    expected = len(DIAMETER_CLASSES) * len(SPEED_CLASSES)
    if len(result) != expected or result["var"].nunique() != expected:
        raise ValueError(
            f"Los metadatos espectrales deben contener {expected} variables únicas; "
            f"se encontraron {result['var'].nunique()}"
        )
    return result


def validate_station_metadata(csv_path: Path) -> None:
    """Validate station 417 coordinates against the downloaded SIATA catalog."""
    if not csv_path.exists():
        return
    stations = pd.read_csv(csv_path)
    code = pd.to_numeric(stations["Codigo"], errors="coerce")
    match = stations.loc[code == int(DISDROMETER_STATION_ID)]
    if len(match) != 1:
        raise ValueError(
            f"El catálogo SIATA debe contener una única estación {DISDROMETER_STATION_ID}"
        )
    station = match.iloc[0]
    if not (
        np.isclose(float(station["latitude"]), DISDROMETER_LAT)
        and np.isclose(float(station["longitude"]), DISDROMETER_LON)
    ):
        raise ValueError("Las coordenadas de la estación 417 no coinciden con el análisis")


# ---------------------------------------------------------------------
# Acumulación incremental de conteos filtrados (var8>0, var14>80)
# ---------------------------------------------------------------------
def aggregate_filtered_counts(
    resultados_dir: Path,
    spectrum_metadata: pd.DataFrame,
    chunk_size: int = disd.DEFAULT_CHUNK_SIZE,
) -> Tuple[
    pd.Series,
    Dict[int, pd.Series],
    Dict[pd.Period, pd.Series],
    Dict[str, object],
]:
    """Acumula los conteos del espectro aplicando los filtros requeridos.

    Returns
    -------
    totals:
        Serie con la suma global por variable del espectro (var42-var481).
    monthly_counts:
        Diccionario mes -> Serie (conteos acumulados del espectro).
    period_counts:
        Diccionario periodo mensual (Period('YYYY-MM')) -> Serie de conteos.
    """
    spec_cols = [col for col in spectrum_metadata["var"] if col.startswith("var")]
    base_series = pd.Series(0.0, index=spec_cols, dtype=np.float64)
    totals = base_series.copy(deep=True)
    monthly_counts: Dict[int, pd.Series] = {m: base_series.copy(deep=True) for m in range(1, 13)}
    period_counts: Dict[pd.Period, pd.Series] = {}

    usecols = [
        "fecha_hora",
        disd.PRECIPITATION_COLUMN,
        disd.QUALITY_COLUMN,
        "var16",
    ] + spec_cols
    quality_summary: Dict[str, object] = {
        "filtered_minutes": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "instrument_reported_particles": 0.0,
        "spectrum_particles": 0.0,
        "negative_spectrum_cells_removed": 0,
        "negative_spectrum_magnitude_removed": 0.0,
        "by_year": {},
    }
    active_year: Optional[int] = None

    for year, chunk in disd.iter_disdro_chunks(
        resultados_dir=resultados_dir,
        precipitation_only=True,
        precipitation_column=disd.PRECIPITATION_COLUMN,
        precipitation_threshold=0.0,
        quality_column=disd.QUALITY_COLUMN,
        quality_threshold=disd.QUALITY_THRESHOLD,
        chunksize=chunk_size,
        usecols=usecols,
    ):
        if year != active_year:
            print(f"Procesando espectros de la estación 417, año {year}...")
            active_year = year
        chunk = chunk.drop(columns=[disd.PRECIPITATION_COLUMN, disd.QUALITY_COLUMN], errors="ignore")
        chunk["fecha_hora"] = pd.to_datetime(chunk["fecha_hora"])
        chunk = chunk.dropna(subset=["fecha_hora"])

        spec_values = chunk[spec_cols].astype(np.float32)
        negative_mask = spec_values < 0
        negative_count = int(negative_mask.sum().sum())
        if negative_count:
            negative_values = spec_values.to_numpy()[negative_mask.to_numpy()]
            negative_magnitude = float(-negative_values.sum(dtype=np.float64))
            quality_summary["negative_spectrum_cells_removed"] = (
                int(quality_summary["negative_spectrum_cells_removed"]) + negative_count
            )
            quality_summary["negative_spectrum_magnitude_removed"] = (
                float(quality_summary["negative_spectrum_magnitude_removed"])
                + negative_magnitude
            )
            # Los conteos de partículas no pueden ser negativos. Estos valores
            # son códigos de ausencia/error del archivo histórico, no mediciones.
            spec_values = spec_values.mask(negative_mask, 0.0)
        chunk_sums = spec_values.sum(axis=0).astype(np.float64)
        totals += chunk_sums

        spectrum_particles = float(chunk_sums.sum())
        instrument_particles = float(chunk["var16"].fillna(0).sum())
        quality_summary["filtered_minutes"] = int(quality_summary["filtered_minutes"]) + len(chunk)
        quality_summary["spectrum_particles"] = (
            float(quality_summary["spectrum_particles"]) + spectrum_particles
        )
        quality_summary["instrument_reported_particles"] = (
            float(quality_summary["instrument_reported_particles"]) + instrument_particles
        )
        first = chunk["fecha_hora"].min()
        last = chunk["fecha_hora"].max()
        if quality_summary["first_timestamp"] is None or first < quality_summary["first_timestamp"]:
            quality_summary["first_timestamp"] = first
        if quality_summary["last_timestamp"] is None or last > quality_summary["last_timestamp"]:
            quality_summary["last_timestamp"] = last

        by_year = quality_summary["by_year"]
        year_summary = by_year.setdefault(
            str(year),
            {"filtered_minutes": 0, "spectrum_particles": 0.0},
        )
        year_summary["filtered_minutes"] += len(chunk)
        year_summary["spectrum_particles"] += spectrum_particles

        month_group = spec_values.groupby(chunk["fecha_hora"].dt.month).sum()
        for mes, series in month_group.iterrows():
            mes = int(mes)
            if mes not in monthly_counts:
                monthly_counts[mes] = base_series.copy(deep=True)
            monthly_counts[mes] += series.astype(np.float64)

        period_group = spec_values.groupby(chunk["fecha_hora"].dt.to_period("M")).sum()
        for period, series in period_group.iterrows():
            period = pd.Period(period, freq="M")
            if period not in period_counts:
                period_counts[period] = base_series.copy(deep=True)
            period_counts[period] += series.astype(np.float64)

        del spec_values

    instrument_total = float(quality_summary["instrument_reported_particles"])
    quality_summary["spectrum_to_instrument_ratio"] = (
        float(quality_summary["spectrum_particles"]) / instrument_total
        if instrument_total > 0
        else np.nan
    )
    for key in ("first_timestamp", "last_timestamp"):
        value = quality_summary[key]
        quality_summary[key] = value.isoformat() if value is not None else None
    print(
        "Control espectral: "
        f"{quality_summary['filtered_minutes']:,} minutos filtrados, "
        f"{quality_summary['spectrum_particles']:,.0f} partículas."
    )
    return totals, monthly_counts, period_counts, quality_summary


# ---------------------------------------------------------------------
# Transformaciones de conteos -> distribuciones
# ---------------------------------------------------------------------
def build_counts_dataframe(series: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    """Mapea los 440 conteos de clases Thies a velocidad y diámetro.

    Cada variable espectral ya es el conteo de una clase individual. Los signos
    ``>`` de los metadatos describen el límite inferior de la clase; no indican
    un conteo acumulativo.
    """
    meta_pivot = (
        meta.pivot(index="speed_min", columns="diameter_min", values="var")
        .sort_index()
        .sort_index(axis=1)
    )
    speeds = meta_pivot.index.to_numpy()
    diameters = meta_pivot.columns.to_numpy()

    expected_shape = (len(SPEED_CLASSES), len(DIAMETER_CLASSES))
    if meta_pivot.shape != expected_shape or meta_pivot.isna().any().any():
        raise ValueError(
            "La cuadrícula espectral debe tener 20 clases de velocidad por "
            "22 clases de diámetro, sin combinaciones faltantes"
        )

    bin_counts = np.zeros((len(speeds), len(diameters)), dtype=np.float64)
    for i, speed in enumerate(speeds):
        for j, diameter in enumerate(diameters):
            var_name = meta_pivot.iloc[i, j]
            value = series.get(var_name, 0.0)
            bin_counts[i, j] = 0.0 if pd.isna(value) else float(value)

    rows = []
    for i, speed in enumerate(speeds):
        for j, diameter in enumerate(diameters):
            count = bin_counts[i, j]
            if count <= 0:
                continue
            rows.append(
                {
                    "speed_min": speed,
                    "speed_center": meta.loc[
                        (meta["speed_min"] == speed)
                        & (meta["diameter_min"] == diameter),
                        "speed_center",
                    ].iloc[0],
                    "diameter_min": diameter,
                    "diameter_center": meta.loc[
                        (meta["speed_min"] == speed)
                        & (meta["diameter_min"] == diameter),
                        "diameter_center",
                    ].iloc[0],
                    "cuentas": count,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "speed_min",
            "speed_center",
            "diameter_min",
            "diameter_center",
            "cuentas",
        ],
    )


def global_distributions(totals: pd.Series, meta: pd.DataFrame):
    counts = build_counts_dataframe(totals, meta)
    if counts.empty:
        raise ValueError("No hay conteos espectrales positivos después de aplicar los filtros")

    diam = (
        counts.groupby("diameter_center", as_index=False)["cuentas"]
        .sum()
        .sort_values("diameter_center")
    )
    vel = (
        counts.groupby("speed_center", as_index=False)["cuentas"]
        .sum()
        .sort_values("speed_center")
    )
    diam["prob"] = diam["cuentas"] / diam["cuentas"].sum()
    vel["prob"] = vel["cuentas"] / vel["cuentas"].sum()

    stats = {
        "gotas": int(counts["cuentas"].sum()),
        "diam_media": weighted_mean(diam["diameter_center"].to_numpy(), diam["cuentas"].to_numpy()),
        "diam_mediana": weighted_median(
            diam["diameter_center"].to_numpy(), diam["cuentas"].to_numpy()
        ),
        "vel_media": weighted_mean(vel["speed_center"].to_numpy(), vel["cuentas"].to_numpy()),
        "vel_mediana": weighted_median(
            vel["speed_center"].to_numpy(), vel["cuentas"].to_numpy()
        ),
    }
    return diam, vel, stats, counts


def monthly_distributions(
    monthly_counts: Dict[int, pd.Series], meta: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for mes, series in monthly_counts.items():
        counts = build_counts_dataframe(series, meta)
        total = counts["cuentas"].sum()
        if counts.empty or total <= 0:
            continue
        counts["prob"] = counts["cuentas"] / total
        out[mes] = counts
    return out


def monthly_bivariate_counts(
    monthly_counts: Dict[int, pd.Series], meta: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for mes, series in monthly_counts.items():
        counts = build_counts_dataframe(series, meta)
        total = counts["cuentas"].sum()
        if counts.empty or total <= 0:
            out[mes] = pd.DataFrame()
            continue
        pivot = counts.pivot_table(
            index="diameter_center", columns="speed_center", values="cuentas", aggfunc="sum", fill_value=0
        )
        out[mes] = pivot
    return out


def period_percentiles(
    period_counts: Dict[pd.Period, pd.Series],
    meta: pd.DataFrame,
    percentiles: Sequence[float],
) -> pd.DataFrame:
    centers = meta.drop_duplicates("var").set_index("var")["diameter_center"]
    result_records = []

    for period, series in sorted(period_counts.items(), key=lambda item: item[0]):
        series = series[series > 0].dropna()
        if series.empty:
            continue
        vals = centers.loc[series.index].to_numpy()
        weights = series.to_numpy()
        q = weighted_percentiles(vals, weights, percentiles)
        record = {"periodo": period.to_timestamp()}
        for p, value in zip(percentiles, q):
            record[f"p{int(p)}"] = value
        record["gotas"] = float(weights.sum())
        result_records.append(record)

    columns = ["periodo", *[f"p{int(p)}" for p in percentiles], "gotas"]
    return pd.DataFrame(result_records, columns=columns).sort_values("periodo")


# ---------------------------------------------------------------------
# Lectura de índices de gran escala (OMEGA, CHI)
# ---------------------------------------------------------------------
def load_index_series(
    nc_path: Path,
    var_name: str,
    level_value: float,
    level_coord: str,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    scale: float = 1.0,
) -> pd.Series:
    with xr.open_dataset(nc_path) as source:
        ds = source
        if "lon" in ds.coords:
            lon = ds["lon"]
            if float(lon.max()) > 180:
                ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
                ds = ds.sortby("lon")

        if var_name not in ds:
            raise KeyError(f"La variable '{var_name}' no existe en '{nc_path.name}'")
        da = ds[var_name]
        fill_value = da.attrs.get("_FillValue", None)
        missing_value = da.attrs.get("missing_value", None)

        if level_coord in da.coords:
            da = da.sel({level_coord: level_value}, method="nearest")
        if "lat" not in da.coords or "lon" not in da.coords:
            raise KeyError(f"'{nc_path.name}' debe contener coordenadas 'lat' y 'lon'")

        lat_coord = da.coords["lat"]
        lon_coord = da.coords["lon"]
        lat_min, lat_max = lat_bounds
        lat_slice = (
            slice(lat_min, lat_max)
            if float(lat_coord[0]) < float(lat_coord[-1])
            else slice(lat_max, lat_min)
        )
        lon_min, lon_max = lon_bounds
        lon_slice = (
            slice(lon_min, lon_max)
            if float(lon_coord[0]) <= float(lon_coord[-1])
            else slice(lon_max, lon_min)
        )
        da = da.sel(lat=lat_slice, lon=lon_slice)
        if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
            raise ValueError(f"El recorte regional quedó vacío para '{nc_path.name}'")

        if fill_value is not None:
            da = da.where(~np.isclose(da, fill_value))
        if missing_value is not None:
            da = da.where(~np.isclose(da, missing_value))

        regional_mean = da.mean(dim=("lat", "lon")).load()
    series = regional_mean.to_series() * scale
    series.index = pd.to_datetime(series.index)
    series.name = var_name
    return series


def load_index_field(
    nc_path: Path,
    var_name: str,
    level_value: float,
    level_coord: str,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
) -> xr.DataArray:
    with xr.open_dataset(nc_path) as source:
        ds = source
        if "lon" in ds.coords:
            lon = ds["lon"]
            if float(lon.max()) > 180:
                ds = ds.assign_coords(lon=((lon + 180) % 360) - 180)
                ds = ds.sortby("lon")

        if var_name not in ds:
            raise KeyError(f"La variable '{var_name}' no existe en '{nc_path.name}'")
        da = ds[var_name]
        fill_value = da.attrs.get("_FillValue", None)
        missing_value = da.attrs.get("missing_value", None)
        if level_coord in da.coords:
            da = da.sel({level_coord: level_value}, method="nearest")
        if "lat" not in da.coords or "lon" not in da.coords:
            raise KeyError(f"'{nc_path.name}' debe contener coordenadas 'lat' y 'lon'")

        lat_coord = da.coords["lat"]
        lon_coord = da.coords["lon"]
        lat_min, lat_max = lat_bounds
        lat_slice = (
            slice(lat_min, lat_max)
            if float(lat_coord[0]) < float(lat_coord[-1])
            else slice(lat_max, lat_min)
        )
        lon_min, lon_max = lon_bounds
        lon_slice = (
            slice(lon_min, lon_max)
            if float(lon_coord[0]) < float(lon_coord[-1])
            else slice(lon_max, lon_min)
        )

        da = da.sel(lat=lat_slice, lon=lon_slice)
        if da.sizes.get("lat", 0) == 0 or da.sizes.get("lon", 0) == 0:
            raise ValueError(f"El recorte regional quedó vacío para '{nc_path.name}'")
        if fill_value is not None:
            da = da.where(~np.isclose(da, fill_value))
        if missing_value is not None:
            da = da.where(~np.isclose(da, missing_value))
        return da.sortby("lat").load()



# ---------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------
def save_figure(fig: plt.Figure, filename: str, dpi: int = 200) -> None:
    """Save and close a figure so the full workflow does not retain it in RAM."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=dpi)
    plt.close(fig)


def safe_axis_limits(lower: float, upper: float) -> Tuple[float, float]:
    """Return non-degenerate limits for sparse one-class distributions."""
    lower = float(lower)
    upper = float(upper)
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.1)
        return lower - padding, upper + padding
    return lower, upper


def annotate_stats(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        1.02,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=STATS_SIZE,
        color="white",
    )


def plot_global_distributions(diam: pd.DataFrame, vel: pd.DataFrame, stats: Dict[str, float]) -> None:
    # Distribución de diámetros
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(diam["diameter_center"], diam["prob"], color=COLOR_DIAM, lw=LINE_WIDTH)
    ax.fill_between(
        diam["diameter_center"],
        0,
        diam["prob"],
        color=COLOR_DIAM,
        alpha=0.2,
    )
    ax.set_title("Distribución global del tamaño de gotas", fontsize=TITLE_SIZE, pad=TITLE_PAD)
    annotate_stats(
        ax,
        f"gotas={stats['gotas']:,} | media={stats['diam_media']:.2f} mm | mediana={stats['diam_mediana']:.2f} mm",
    )
    ax.set_xlabel("Diámetro (mm)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Probabilidad de ocurrencia", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "01_distribucion_global_diametros.png")

    # Distribución de velocidades
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(vel["speed_center"], vel["prob"], color=COLOR_VEL, lw=LINE_WIDTH)
    ax.fill_between(
        vel["speed_center"],
        0,
        vel["prob"],
        color=COLOR_VEL,
        alpha=0.2,
    )
    ax.set_title("Distribución global de velocidades de caída", fontsize=TITLE_SIZE, pad=TITLE_PAD)
    annotate_stats(
        ax,
        f"gotas={stats['gotas']:,} | media={stats['vel_media']:.2f} m/s | mediana={stats['vel_mediana']:.2f} m/s",
    )
    ax.set_xlabel("Velocidad (m/s)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Probabilidad de ocurrencia", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "02_distribucion_global_velocidades.png")



def plot_monthly_distributions(
    distributions: Dict[int, pd.DataFrame],
    global_distribution: pd.DataFrame,
    value_col: str,
    title: str,
    line_color: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=False, sharey=False)
    fig.suptitle(title, fontsize=TITLE_SIZE + 4, y=0.94)

    if value_col == "diameter_center":
        global_x = global_distribution["diameter_center"].to_numpy()
    else:
        global_x = global_distribution["speed_center"].to_numpy()
    global_y = global_distribution["prob"].to_numpy()

    monthly_curves: Dict[int, Dict[str, np.ndarray]] = {}
    y_max = float(global_y.max() if global_y.size else 0.0)

    for month, data in distributions.items():
        if data is None or data.empty:
            monthly_curves[month] = {}
            continue

        agg = (
            data.groupby(value_col)["cuentas"]
            .sum()
            .sort_index()
        )
        total = agg.sum()
        if total <= 0:
            monthly_curves[month] = {}
            continue

        probs = agg / total
        y_max = max(y_max, float(probs.max()))
        monthly_curves[month] = {
            "x": agg.index.to_numpy(),
            "y": probs.values,
            "n": int(data["cuentas"].sum()),
            "mean": weighted_mean(agg.index.to_numpy(), agg.values),
        }

    y_max = y_max * 1.05 if y_max > 0 else 0.05

    for idx, month in enumerate(range(1, 13)):
        ax = axes.flatten()[idx]
        curve = monthly_curves.get(month, {})

        if not curve:
            ax.set_facecolor("black")
            ax.plot(global_x, global_y, color=line_color, lw=LINE_WIDTH, ls="--", alpha=0.9)
            ax.text(
                0.5,
                0.5,
                "Sin datos",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=LEGEND_SIZE,
            )
            ax.set_title(f"{month:02d}", fontsize=LABEL_SIZE)
            ax.set_ylim(0, y_max)
            continue

        ax.fill_between(
            curve["x"],
            curve["y"],
            color=line_color,
            alpha=0.25,
        )
        ax.plot(
            curve["x"],
            curve["y"],
            color=line_color,
            lw=LINE_WIDTH,
        )
        ax.plot(
            global_x,
            global_y,
            color="#ebdd22",
            lw=LINE_WIDTH,
            ls="--",
            alpha=0.9,
        )

        ax.set_title(f"{month:02d}", fontsize=LABEL_SIZE)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, y_max)
        ax.text(
            0.98,
            0.95,
            f"n_gotas={curve['n']:,}\nmedia={curve['mean']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LEGEND_SIZE - 1,
            bbox=dict(facecolor="black", alpha=0.6, pad=0.4),
        )

    x_label = "Diámetro (mm)" if value_col == "diameter_center" else "Velocidad (m/s)"
    for ax in axes[-1]:
        ax.set_xlabel(x_label, fontsize=LABEL_SIZE)
    for row in axes:
        for axis in row:
            axis.set_ylabel("Probabilidad", fontsize=LABEL_SIZE)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    save_figure(fig, filename)


def build_bivariate_pivot(counts: pd.DataFrame) -> pd.DataFrame:
    return (
        counts.pivot_table(
            index="diameter_center",
            columns="speed_center",
            values="cuentas",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )


def plot_bivariate_global(pivot: pd.DataFrame) -> None:
    if pivot.empty:
        return

    prob = pivot / pivot.values.sum()
    vmax = float(prob.values.max())
    vmax = vmax if vmax > 0 else 1e-6
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 7))
    diam_limits = safe_axis_limits(pivot.index.min(), pivot.index.max())
    speed_limits = safe_axis_limits(pivot.columns.min(), pivot.columns.max())
    img = ax.imshow(
        prob.values.T,
        origin="lower",
        aspect="auto",
        extent=[
            *diam_limits,
            *speed_limits,
        ],
        cmap=GLOBAL_CMAP,
        norm=norm,
    )
    ax.set_title("Histograma bivariado global", fontsize=TITLE_SIZE)
    ax.set_xlabel("Diámetro (mm)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Velocidad (m/s)", fontsize=LABEL_SIZE)
    ax.grid(False)

    gotas = int(pivot.values.sum())
    diameters = pivot.sum(axis=1)
    d50 = weighted_percentiles(
        diameters.index.to_numpy(), diameters.to_numpy(), [50]
    )[0]
    ax.text(
        0.98,
        0.95,
        f"gotas={gotas:,}\nD50={d50:.2f} mm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=LEGEND_SIZE - 1,
        bbox=dict(facecolor="black", alpha=0.6, pad=0.3),
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.047, pad=0.04)
    cbar.ax.tick_params(color="white", labelcolor="white")
    cbar.set_label("Probabilidad relativa", color="white", fontsize=LABEL_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "03_histograma_bivariado_global.png")

def plot_bivariate_reference(counts: pd.DataFrame) -> None:
    pivot = (
        counts.pivot_table(
            index="diameter_center",
            columns="speed_center",
            values="cuentas",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return

    prob = pivot / pivot.values.sum()
    data = zoom(prob.values, zoom=2, order=1)
    positive = data[data > 0]
    vmin = positive.min() if positive.size else 1e-6
    vmax = positive.max() if positive.size else 1.0
    threshold = max(vmin, 1e-4)
    masked = np.ma.array(data, mask=data <= threshold)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    speed_limits = safe_axis_limits(pivot.columns.min(), pivot.columns.max())
    diam_limits = safe_axis_limits(pivot.index.min(), pivot.index.max())
    img = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=[
            *speed_limits,
            *diam_limits,
        ],
        cmap="turbo",
        norm=LogNorm(vmin=threshold, vmax=vmax),
    )

    diam_range = np.linspace(pivot.index.min(), pivot.index.max(), 400)
    vel_curve = 9.65 - 10.3 * np.exp(-0.6 * diam_range)
    ax.plot(vel_curve, diam_range, color="#ff4a4a", lw=3, label="Atlas et al. 1973")
    band_offset = 1.5
    ax.plot(
        np.maximum(0.2, vel_curve - band_offset),
        diam_range,
        color="#ff9a9a",
        lw=1.5,
        ls="--",
        alpha=0.6,
    )
    ax.plot(
        vel_curve + band_offset,
        diam_range,
        color="#ff9a9a",
        lw=1.5,
        ls="--",
        alpha=0.6,
    )

    diameters = pivot.index.to_numpy()
    mode_speeds = []
    for _, row in prob.iterrows():
        if row.sum() <= 0:
            mode_speeds.append(np.nan)
        else:
            mode_speeds.append(row.idxmax())

    ax.scatter(
        mode_speeds,
        diameters,
        marker="*",
        s=70,
        color="#ffd700",
        edgecolor="black",
        linewidth=0.5,
        label="Máxima probabilidad por diámetro",
    )

    ax.set_title(
        "Histograma bivariado con curva de referencia",
        fontsize=TITLE_SIZE,
        pad=TITLE_PAD,
    )
    ax.set_xlabel("Velocidad (m/s)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Diámetro (mm)", fontsize=LABEL_SIZE)
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, facecolor="#111111")

    cbar = fig.colorbar(img, ax=ax, fraction=0.05, pad=0.02)
    cbar.ax.tick_params(color="white", labelcolor="white")
    cbar.set_label("Probabilidad relativa (escala log)", color="white", fontsize=LABEL_SIZE)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "11_histograma_bivariado_referencia.png")


def plot_bivariate_monthly(
    monthly_pivots: Dict[int, pd.DataFrame],
    global_pivot: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    fig.suptitle("Histogramas bivariados por mes", fontsize=TITLE_SIZE + 4, y=0.94)

    global_prob = global_pivot / global_pivot.values.sum()
    diff_data: Dict[int, Optional[pd.DataFrame]] = {}
    max_abs = 0.0

    for month, pivot in monthly_pivots.items():
        if pivot is None or pivot.empty:
            diff_data[month] = None
            continue

        month_prob = pivot / pivot.values.sum()
        month_aligned = (
            month_prob.reindex(global_prob.index, fill_value=0.0)
            .reindex(global_prob.columns, axis=1, fill_value=0.0)
        )
        diff = month_aligned - global_prob
        diff_data[month] = diff
        if not diff.empty:
            max_abs = max(max_abs, float(np.nanmax(np.abs(diff.values))))

    max_abs = max_abs if max_abs > 0 else 1e-6
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    for idx, month in enumerate(range(1, 13)):
        ax = axes.flatten()[idx]
        diff = diff_data.get(month)
        pivot = monthly_pivots.get(month)
        if diff is None or diff.empty or pivot is None or pivot.empty:
            ax.set_facecolor("black")
            ax.text(
                0.5,
                0.5,
                "Sin datos",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=LEGEND_SIZE,
            )
            ax.set_title(f"{month:02d}", fontsize=LABEL_SIZE)
            continue

        diam_limits = safe_axis_limits(diff.index.min(), diff.index.max())
        speed_limits = safe_axis_limits(diff.columns.min(), diff.columns.max())
        ax.imshow(
            diff.values.T,
            origin="lower",
            aspect="auto",
            extent=[
                *diam_limits,
                *speed_limits,
            ],
            cmap=DIV_CMAP,
            norm=norm,
        )
        ax.set_title(f"{month:02d}", fontsize=LABEL_SIZE)
        ax.set_xlabel("Diámetro (mm)", fontsize=LABEL_SIZE - 2)
        ax.set_ylabel("Velocidad (m/s)", fontsize=LABEL_SIZE - 2)

        gotas = int(pivot.values.sum())
        diameters = pivot.sum(axis=1)
        d50 = weighted_percentiles(
            diameters.index.to_numpy(), diameters.to_numpy(), [50]
        )[0]
        ax.text(
            0.98,
            0.95,
            f"gotas={gotas:,}\nD50={d50:.2f} mm",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=LEGEND_SIZE - 1,
            bbox=dict(facecolor="black", alpha=0.6, pad=0.3),
        )

    fig.tight_layout(rect=[0, 0.02, 0.92, 0.95])
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=DIV_CMAP),
        ax=axes.ravel().tolist(),
        fraction=0.015,
        pad=0.01,
    )
    cbar.set_label(
        "Diferencia de probabilidad (mes - global)",
        color="white",
        fontsize=LABEL_SIZE,
    )
    cbar.ax.tick_params(color="white", labelcolor="white")
    save_figure(fig, "04_histogramas_bivariados_por_mes.png")

def plot_percentile_timeseries(percentiles_df: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(14, 6))
    for key, color in PERCENTILE_COLORS.items():
        if key not in percentiles_df.columns:
            continue
        ax.plot(
            percentiles_df["periodo"],
            percentiles_df[key],
            marker="o",
            ms=4,
            lw=LINE_WIDTH,
            color=color,
            label=key.upper(),
            alpha=0.9,
        )
    ax.set_title("Percentiles mensuales del tamaño de gotas", fontsize=TITLE_SIZE)
    ax.set_xlabel("Fecha (mes)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Diámetro (mm)", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEGEND_SIZE, ncol=1, loc="upper left", facecolor="#111111")
    fig.tight_layout()
    save_figure(fig, "05_series_percentiles_diametro.png")
    return percentiles_df.set_index("periodo")


def scatter_stats(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[int, float, float, float]:
    valid = df[[x_col, y_col]].dropna()
    if len(valid) < 2 or valid[x_col].nunique() < 2:
        return 0, np.nan, np.nan, np.nan
    n = len(valid)
    r = valid.corr().iloc[0, 1]
    slope, intercept = np.polyfit(valid[x_col], valid[y_col], 1)
    r2 = r**2 if not np.isnan(r) else np.nan
    return n, r, r2, slope


def plot_scatter_percentiles_vs_index(
    monthly_percentiles: pd.DataFrame,
    index_series: pd.Series,
    index_name: str,
    units: str,
    title: str,
    beta_scale: float = 1.0,
    beta_suffix: str = "",
    filename: str = "scatter.png",
) -> None:
    df = monthly_percentiles.join(index_series.rename(index_name), how="inner")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color in PERCENTILE_COLORS.items():
        if key not in df.columns:
            continue
        ax.scatter(
            df[index_name],
            df[key],
            color=color,
            s=40,
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            label=key.upper(),
        )
        ax.plot(
            df[index_name],
            df[key],
            color=color,
            alpha=0.15,
            lw=1.0,
        )

    n, r, r2, slope = scatter_stats(df, index_name, "p50")
    slope_scaled = slope * beta_scale if slope is not None else np.nan
    stats_text = (
        f"n={n} | r(P50)={r:.2f} | R²={r2:.2f} | β={slope_scaled:.3e}{beta_suffix}"
        if n > 0
        else "Sin datos suficientes"
    )

    annotate_stats(ax, stats_text)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.set_xlabel(f"{index_name} ({units})", fontsize=LABEL_SIZE)
    ax.set_ylabel("Diámetro (mm)", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right", facecolor="#111111")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_scatter_omega_vs_chi(
    omega: pd.Series, chi: pd.Series, title: str, filename: str
) -> None:
    df = pd.concat([omega.rename("OMEGA"), chi.rename("CHI")], axis=1).dropna()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        df["OMEGA"],
        df["CHI"],
        c=np.arange(len(df)),
        cmap="plasma",
        s=40,
        alpha=0.9,
        edgecolors="none",
    )
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_xlabel("OMEGA (Pa/s)", fontsize=LABEL_SIZE)
    ax.set_ylabel("CHI (m²/s)", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("Índice temporal (meses)", color="white", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(color="white", labelcolor="white")

    fig.tight_layout()
    save_figure(fig, filename)


def compute_correlation_map(
    field: xr.DataArray, target_lat: float, target_lon: float
) -> Optional[Tuple[xr.DataArray, float, float, int]]:
    """Compute the Pearson correlation map against the disdrometer pixel."""
    if "time" not in field.dims:
        return None

    point = field.sel(lat=target_lat, lon=target_lon, method="nearest")
    valid_times = point["time"].where(point.notnull(), drop=True)
    if valid_times.size < 3:
        return None

    valid_times = pd.to_datetime(valid_times.values)
    field = field.sel(time=valid_times)
    point = point.sel(time=valid_times)
    corr = xr.corr(field, point, dim="time")
    corr = corr.sortby("lat")
    ref_lat = float(point.coords["lat"].values)
    ref_lon = float(point.coords["lon"].values)
    n_samples = int(valid_times.size)
    return corr, ref_lat, ref_lon, n_samples


def plot_correlation_map(
    corr_da: xr.DataArray,
    ref_lat: float,
    ref_lon: float,
    n_samples: int,
    title: str,
    filename: str,
) -> None:
    """Plot the spatial correlation map."""
    if corr_da is None:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    lon_edges = coordinate_cell_edges(corr_da["lon"].values, REGION_LON_BOUNDS)
    lat_edges = coordinate_cell_edges(corr_da["lat"].values, REGION_LAT_BOUNDS)
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        corr_da,
        cmap=DIV_CMAP,
        norm=norm,
        shading="flat",
    )
    ax.scatter(
        ref_lon,
        ref_lat,
        marker="*",
        s=140,
        color="white",
        edgecolor="#111111",
        linewidths=0.8,
        zorder=5,
        label="Disdrómetro 417",
    )
    stats_text = f"n meses = {n_samples}\nPixel ref.: ({ref_lat:.2f}°, {ref_lon:.2f}°)"
    ax.text(
        0.98,
        0.02,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=STATS_SIZE,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", boxstyle="round,pad=0.3"),
    )
    ax.set_xlabel("Longitud", fontsize=LABEL_SIZE)
    ax.set_ylabel("Latitud", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.set_xlim(REGION_LON_BOUNDS)
    ax.set_ylim(REGION_LAT_BOUNDS)
    ax.grid(True, linestyle="--", alpha=0.25)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación de Pearson", fontsize=LABEL_SIZE, color="white")
    cbar.ax.tick_params(color="white", labelcolor="white")

    fig.tight_layout()
    save_figure(fig, filename)


def coordinate_cell_edges(
    coordinates: Sequence[float], bounds: Tuple[float, float]
) -> np.ndarray:
    """Return visible grid-cell edges without inventing extra grid points.

    A one-coordinate NOAA subset still represents a finite cell inside the
    requested geographic box. ``pcolormesh`` needs its two outer edges to draw
    that cell; for larger subsets, only midpoints between original coordinates
    are added.
    """
    values = np.asarray(coordinates, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("No se pueden construir bordes para una coordenada vacía")
    lower, upper = sorted(map(float, bounds))
    if values.size == 1:
        return np.array([lower, upper], dtype=float)
    if np.any(np.diff(values) <= 0):
        raise ValueError("Las coordenadas deben estar ordenadas y no repetidas")
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(([lower], midpoints, [upper]))


def correlate_field_with_series(
    field: xr.DataArray, reference: pd.Series
) -> Optional[Tuple[xr.DataArray, int]]:
    """Correlate a gridded field with a reference time series."""
    if field is None or reference is None:
        return None

    series = reference.dropna().copy()
    if series.empty:
        return None
    if isinstance(series.index, pd.PeriodIndex):
        series.index = series.index.to_timestamp()
    series.index = pd.to_datetime(series.index)
    series.index.name = "time"

    field = field.assign_coords(time=pd.to_datetime(field["time"].values))
    ref_da = xr.DataArray(series.values, coords={"time": series.index}, dims="time")
    aligned_field, aligned_ref = xr.align(field, ref_da, join="inner")
    n_samples = int(aligned_field.sizes.get("time", 0))
    if n_samples < 3:
        return None

    corr = xr.corr(aligned_field, aligned_ref, dim="time")
    return corr.sortby("lat"), n_samples


def plot_percentile_correlation_maps(
    entries: Sequence[Tuple[str, xr.DataArray, int]],
    ref_lat: float,
    ref_lon: float,
    filename: str,
) -> None:
    """Plot multi-panel correlation maps between indices and percentiles."""
    valid_entries = [entry for entry in entries if entry[1] is not None]
    if not valid_entries:
        return

    n_panels = len(valid_entries)
    ncols = 2
    nrows = math.ceil(n_panels / ncols)
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs else {}
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 5.3 * nrows),
        subplot_kw=subplot_kw,
    )
    axes = np.atleast_1d(axes).flatten()
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    meshes = []

    for ax, (title, da, n_samples) in zip(axes, valid_entries):
        extent = [
            float(REGION_LON_BOUNDS[0]),
            float(REGION_LON_BOUNDS[1]),
            float(REGION_LAT_BOUNDS[0]),
            float(REGION_LAT_BOUNDS[1]),
        ]
        transform = ccrs.PlateCarree() if ccrs else None
        if ccrs:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(color="#cccccc", linewidth=0.6)
            if cfeature is not None:
                ax.add_feature(
                    cfeature.BORDERS,
                    linewidth=0.4,
                    edgecolor="#999999",
                )
        lon_edges = coordinate_cell_edges(da["lon"].values, REGION_LON_BOUNDS)
        lat_edges = coordinate_cell_edges(da["lat"].values, REGION_LAT_BOUNDS)
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            da,
            cmap=DIV_CMAP,
            norm=norm,
            shading="flat",
            transform=transform,
        )
        ax.scatter(
            ref_lon,
            ref_lat,
            marker="*",
            s=120,
            color="white",
            edgecolor="#111111",
            linewidths=0.8,
            transform=transform,
            zorder=5,
        )
        ax.set_title(title, fontsize=TITLE_SIZE - 2)
        ax.set_xlabel("Longitud", fontsize=LABEL_SIZE)
        ax.set_ylabel("Latitud", fontsize=LABEL_SIZE)
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.text(
            0.02,
            0.02,
            f"n = {n_samples} meses",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=STATS_SIZE,
            color="white",
            bbox=dict(
                facecolor="black",
                alpha=0.35,
                boxstyle="round,pad=0.3",
                edgecolor="none",
            ),
        )
        meshes.append(mesh)

    for ax in axes[len(valid_entries) :]:
        ax.remove()

    cbar_ax = fig.add_axes([0.27, 0.055, 0.46, 0.025])
    cbar = fig.colorbar(meshes[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Correlación de Pearson", fontsize=LABEL_SIZE, color="white")
    cbar.ax.tick_params(color="white", labelcolor="white")

    fig.suptitle(
        "Correlaciones espaciales (percentiles vs índices)",
        fontsize=TITLE_SIZE + 2,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.96,
        bottom=0.17,
        top=0.90,
        hspace=0.30,
        wspace=0.22,
    )
    save_figure(fig, filename, dpi=220)


# ---------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------
def main() -> None:
    resultados_dir = BASE_DIR / "resultados"
    metadata_csv = BASE_DIR / "varsDisdroBD_names_corrected.csv"
    station_metadata_csv = SOURCE_DATA_DIR / "siata" / "Metadata_Disdros.csv"
    omega_nc = SOURCE_DATA_DIR / "noaa" / "omega.mon.mean.nc"
    chi_nc = SOURCE_DATA_DIR / "noaa" / "chi.mon.mean.nc"

    validate_station_metadata(station_metadata_csv)

    if not resultados_dir.exists():
        raise FileNotFoundError(
            "Falta la carpeta de entrada 'resultados'. Consulta README.md para "
            "la estructura reproducible esperada."
        )
    if not disd.list_disdro_files(resultados_dir=resultados_dir):
        raise FileNotFoundError(
            "No se encontraron archivos resultados/Disdrodata_417_YYYY.csv"
        )

    if metadata_csv.exists():
        spectrum_meta = load_spectrum_metadata(metadata_csv)
    else:
        spectrum_meta = build_default_spectrum_metadata()
        print(
            "Aviso: no se encontró varsDisdroBD_names_corrected.csv; "
            "se usarán las 440 clases canónicas del Thies LPM."
        )
    totals, monthly_counts, period_counts, quality_summary = aggregate_filtered_counts(
        resultados_dir, spectrum_meta, chunk_size=10_000
    )

    # DISTRIBUCIONES GLOBALES
    diam, vel, stats, counts_long = global_distributions(totals, spectrum_meta)
    plot_global_distributions(diam, vel, stats)

    # DISTRIBUCIONES MENSUALES
    monthly_dist = monthly_distributions(monthly_counts, spectrum_meta)
    plot_monthly_distributions(
        monthly_dist,
        diam,
        "diameter_center",
        "Distribución mensual del tamaño de gotas",
        COLOR_DIAM,
        "06_distribuciones_mensuales_diametros.png",
    )
    plot_monthly_distributions(
        monthly_dist,
        vel,
        "speed_center",
        "Distribución mensual de las velocidades de caída",
        COLOR_VEL,
        "07_distribuciones_mensuales_velocidades.png",
    )

    # HISTOGRAMAS BIVARIADOS
    global_pivot = build_bivariate_pivot(counts_long)
    plot_bivariate_global(global_pivot)
    monthly_pivots = monthly_bivariate_counts(monthly_counts, spectrum_meta)
    plot_bivariate_monthly(monthly_pivots, global_pivot)
    plot_bivariate_reference(counts_long)

    # PERCENTILES MENSUALES
    percentiles = period_percentiles(period_counts, spectrum_meta, percentiles=[10, 25, 50, 75, 90])
    monthly_percentiles = plot_percentile_timeseries(percentiles)

    # RESULTADOS NUMÉRICOS AUDITABLES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diam.to_csv(OUTPUT_DIR / "distribucion_global_diametros.csv", index=False)
    vel.to_csv(OUTPUT_DIR / "distribucion_global_velocidades.csv", index=False)
    percentiles.to_csv(OUTPUT_DIR / "percentiles_mensuales.csv", index=False)
    summary = {
        "station": {
            "id": int(DISDROMETER_STATION_ID),
            "name": DISDROMETER_STATION_NAME,
            "latitude": DISDROMETER_LAT,
            "longitude": DISDROMETER_LON,
        },
        "geographic_box": {
            "latitude_bounds": list(REGION_LAT_BOUNDS),
            "longitude_bounds": list(REGION_LON_BOUNDS),
        },
        "filters": {
            "precipitation": f"{disd.PRECIPITATION_COLUMN} > 0",
            "quality": f"{disd.QUALITY_COLUMN} > {disd.QUALITY_THRESHOLD}",
        },
        "global_statistics": stats,
        "quality_control": quality_summary,
    }
    summary_path = OUTPUT_DIR / "resumen_resultados.json"

    def write_summary() -> None:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    write_summary()

    missing_netcdf = [path.name for path in (omega_nc, chi_nc) if not path.exists()]
    if missing_netcdf:
        print(
            "Aviso: se omitieron los análisis OMEGA/CHI porque faltan: "
            + ", ".join(missing_netcdf)
        )
        print(f"Figuras del disdrómetro guardadas en: {OUTPUT_DIR.resolve()}")
        return

    # LECTURA DE ÍNDICES OMEGA & CHI
    lat_bounds = REGION_LAT_BOUNDS
    lon_bounds = REGION_LON_BOUNDS

    omega_series = load_index_series(
        omega_nc, "omega", 500.0, "level", lat_bounds, lon_bounds, scale=1.0
    )
    chi_series = load_index_series(
        chi_nc, "chi", 0.2101, "level", lat_bounds, lon_bounds, scale=1.0
    )
    omega_field = load_index_field(
        omega_nc, "omega", 500.0, "level", lat_bounds, lon_bounds
    )
    chi_field = load_index_field(
        chi_nc, "chi", 0.2101, "level", lat_bounds, lon_bounds
    )

    summary["noaa_reanalysis"] = {
        "omega": {
            "file": str(omega_nc.relative_to(BASE_DIR)).replace("\\", "/"),
            "selected_level_hpa": float(omega_field["level"].item()),
            "units": omega_field.attrs.get("units", "Pa/s"),
            "time_start": pd.Timestamp(omega_field["time"].min().item()).isoformat(),
            "time_end": pd.Timestamp(omega_field["time"].max().item()).isoformat(),
            "regional_grid": {
                "latitudes": int(omega_field.sizes["lat"]),
                "longitudes": int(omega_field.sizes["lon"]),
            },
        },
        "chi": {
            "file": str(chi_nc.relative_to(BASE_DIR)).replace("\\", "/"),
            "selected_sigma_level": float(chi_field["level"].item()),
            "units": chi_field.attrs.get("units", "m²/s"),
            "time_start": pd.Timestamp(chi_field["time"].min().item()).isoformat(),
            "time_end": pd.Timestamp(chi_field["time"].max().item()).isoformat(),
            "regional_grid": {
                "latitudes": int(chi_field.sizes["lat"]),
                "longitudes": int(chi_field.sizes["lon"]),
            },
        },
    }

    # Convertir a PeriodIndex mensual
    omega_monthly = omega_series.resample("MS").mean().to_period("M")
    chi_monthly = chi_series.resample("MS").mean().to_period("M")
    monthly_percentiles_period = monthly_percentiles.copy()
    monthly_percentiles_period.index = monthly_percentiles_period.index.to_period("M")

    correlation_rows: List[Dict[str, object]] = []
    for index_name, index_values in (("CHI", chi_monthly), ("OMEGA", omega_monthly)):
        joined = monthly_percentiles_period.join(
            index_values.rename(index_name), how="inner"
        )
        for percentile_name in PERCENTILE_COLORS:
            if percentile_name not in joined:
                continue
            n, r, r2, slope = scatter_stats(joined, index_name, percentile_name)
            correlation_rows.append(
                {
                    "x": index_name,
                    "y": percentile_name.upper(),
                    "n": int(n),
                    "pearson_r": float(r) if np.isfinite(r) else np.nan,
                    "r_squared": float(r2) if np.isfinite(r2) else np.nan,
                    "linear_slope_y_per_x_unit": (
                        float(slope) if np.isfinite(slope) else np.nan
                    ),
                }
            )

    omega_chi = pd.concat(
        [omega_monthly.rename("OMEGA"), chi_monthly.rename("CHI")], axis=1
    ).dropna()
    n, r, r2, slope = scatter_stats(omega_chi, "OMEGA", "CHI")
    correlation_rows.append(
        {
            "x": "OMEGA",
            "y": "CHI",
            "n": int(n),
            "pearson_r": float(r) if np.isfinite(r) else np.nan,
            "r_squared": float(r2) if np.isfinite(r2) else np.nan,
            "linear_slope_y_per_x_unit": float(slope) if np.isfinite(slope) else np.nan,
        }
    )
    pd.DataFrame(correlation_rows).to_csv(
        OUTPUT_DIR / "correlaciones_mensuales.csv", index=False
    )
    summary["monthly_correlations_file"] = "figuras/correlaciones_mensuales.csv"
    write_summary()

    # FIGURAS DE DISPERSIÓN
    plot_scatter_percentiles_vs_index(
        monthly_percentiles_period,
        chi_monthly,
        "CHI",
        "m²/s",
        "Percentiles de Diametro vs CHI - Promedio mensual regional",
        beta_scale=1e6,
        beta_suffix=" mm / (10⁶ m² s⁻¹)",
        filename="08_percentiles_vs_chi.png",
    )

    plot_scatter_percentiles_vs_index(
        monthly_percentiles_period,
        omega_monthly,
        "OMEGA",
        "Pa/s",
        "Percentiles de Diametro vs OMEGA - Promedio mensual regional",
        beta_scale=1.0,
        beta_suffix=" mm / (Pa s⁻¹)",
        filename="09_percentiles_vs_omega.png",
    )

    plot_scatter_omega_vs_chi(
        omega_monthly,
        chi_monthly,
        "OMEGA vs CHI - Promedio regional",
        "10_omega_vs_chi.png",
    )

    # MAPAS DE CORRELACIÓN ESPACIAL
    if not monthly_percentiles.empty:
        time_start = monthly_percentiles.index.min()
        time_end = monthly_percentiles.index.max()
        start_str = time_start.strftime("%Y-%m-01")
        end_str = (time_end + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        omega_field = omega_field.sel(time=slice(start_str, end_str))
        chi_field = chi_field.sel(time=slice(start_str, end_str))

    omega_corr = compute_correlation_map(omega_field, DISDROMETER_LAT, DISDROMETER_LON)
    if omega_corr is not None:
        corr_da, ref_lat, ref_lon, n_samples = omega_corr
        plot_correlation_map(
            corr_da,
            ref_lat,
            ref_lon,
            n_samples,
            "Correlación espacial de OMEGA (500 hPa)",
            "12_correlacion_omega.png",
        )

    chi_corr = compute_correlation_map(chi_field, DISDROMETER_LAT, DISDROMETER_LON)
    if chi_corr is not None:
        corr_da, ref_lat, ref_lon, n_samples = chi_corr
        plot_correlation_map(
            corr_da,
            ref_lat,
            ref_lon,
            n_samples,
            "Correlación espacial de CHI (0.21 σ)",
            "13_correlacion_chi.png",
        )

    percentile_series_df = monthly_percentiles.copy().sort_index()
    percentile_series_df.index = pd.to_datetime(percentile_series_df.index)
    percentile_entries: List[Tuple[str, xr.DataArray, int]] = []
    for column, label, field_da in [
        ("p50", "CHI vs P50", chi_field),
        ("p90", "CHI vs P90", chi_field),
        ("p50", "OMEGA vs P50", omega_field),
        ("p90", "OMEGA vs P90", omega_field),
    ]:
        if field_da is None or column not in percentile_series_df.columns:
            continue
        result = correlate_field_with_series(field_da, percentile_series_df[column])
        if result is None:
            continue
        corr_da, n_samples = result
        percentile_entries.append((label, corr_da, n_samples))

    plot_percentile_correlation_maps(
        percentile_entries,
        DISDROMETER_LAT,
        DISDROMETER_LON,
        "14_correlaciones_percentiles_indices.png",
    )

    print(f"Figuras guardadas en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"No se puede ejecutar el análisis: {exc}") from None
