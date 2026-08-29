from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import pandas as pd

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "resultados"
DEFAULT_STATION_ID = "417"
PRECIPITATION_COLUMN = "var8"
PRECIPITATION_SPECTRUM_START = 42
PRECIPITATION_SPECTRUM_END = 481  # inclusive
PRECIPITATION_SPECTRUM_COLUMNS = [
    f"var{i}" for i in range(PRECIPITATION_SPECTRUM_START, PRECIPITATION_SPECTRUM_END + 1)
]
NO_PRECIPITATION_TOKENS = {"np", "NP", "Np", "nP"}
ERROR_SENTINELS: Iterable[float] = (
    99.0,
    99.9,
    999.0,
    999.9,
    9999.0,
    9999.9,
    99999.0,
    99999.9,
)
DEFAULT_CHUNK_SIZE = 10_000
QUALITY_COLUMN = "var14"
QUALITY_THRESHOLD = 80.0

__all__ = [
    "parse_station_year",
    "list_disdro_files",
    "read_disdro_file",
    "iter_disdro_chunks",
    "iter_disdro_files",
    "read_all_disdro_files",
    "PRECIPITATION_COLUMN",
    "PRECIPITATION_SPECTRUM_COLUMNS",
    "DEFAULT_CHUNK_SIZE",
    "QUALITY_COLUMN",
    "QUALITY_THRESHOLD",
]


def parse_station_year(file_path: Path) -> Tuple[str, int]:
    """Return the (station_id, year) encoded in a disdrometer CSV filename."""
    stem = file_path.stem  # e.g. "Disdrodata_417_2019"
    parts = stem.split("_")
    if len(parts) != 3 or parts[0].lower() != "disdrodata":
        raise ValueError(
            f"File name '{file_path.name}' does not match the expected pattern"
        )
    station_id, year_text = parts[1], parts[2]
    try:
        year = int(year_text)
    except ValueError as exc:
        raise ValueError(f"Year segment '{year_text}' is not an integer") from exc
    return station_id, year


def list_disdro_files(
    resultados_dir: Path = DEFAULT_RESULTS_DIR,
    station_id: Optional[str] = DEFAULT_STATION_ID,
) -> List[Path]:
    """List the available disdrometer CSV files ordered by year."""
    resultados_dir = Path(resultados_dir)
    if not resultados_dir.exists():
        raise FileNotFoundError(
            f"Directory '{resultados_dir}' does not exist"
        )

    candidates: List[Tuple[int, Path]] = []
    for path in resultados_dir.glob("Disdrodata_*_*.csv"):
        current_station, year = parse_station_year(path)
        if station_id is None or current_station == str(station_id):
            candidates.append((year, path))

    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def _build_read_csv_kwargs(overrides: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "sep": ",",
        "decimal": ".",
        "encoding": "latin-1",
        "low_memory": False,
    }
    defaults.update(overrides)
    return defaults


def read_disdro_file(
    file_path: Path,
    *,
    error_sentinels: Optional[Iterable[float]] = None,
    sentinel_columns: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a single disdrometer CSV file into a pandas DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' was not found")

    frame = pd.read_csv(file_path, **_build_read_csv_kwargs(read_csv_kwargs))
    return _normalize_disdro_frame(
        frame,
        error_sentinels=error_sentinels,
        sentinel_columns=sentinel_columns,
    )


def _normalize_disdro_frame(
    frame: pd.DataFrame,
    *,
    error_sentinels: Optional[Iterable[float]] = None,
    sentinel_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Format timestamp, convert numeric fields and clean sentinel values."""
    frame = frame.copy()

    if "fecha_hora" in frame.columns:
        fechas = frame["fecha_hora"].astype(str).str.strip()
        mask_semicolon = fechas.str.contains(";", regex=False)
        parsed = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
        if mask_semicolon.any():
            parsed_semicolon = pd.to_datetime(
                fechas.loc[mask_semicolon],
                format="%d.%m.%y;%H:%M:%S",
                errors="coerce",
            )
            parsed.loc[mask_semicolon] = parsed_semicolon
        if (~mask_semicolon).any():
            parsed.loc[~mask_semicolon] = pd.to_datetime(
                fechas.loc[~mask_semicolon], errors="coerce"
            )
        frame["fecha_hora"] = parsed

    if "cliente" in frame.columns:
        frame["cliente"] = frame["cliente"].astype(str).str.strip()

    numeric_columns = [
        col for col in frame.columns if col not in {"fecha_hora", "cliente"}
    ]
    if numeric_columns:
        frame[numeric_columns] = (
            frame[numeric_columns]
            .replace(list(NO_PRECIPITATION_TOKENS), 0)
            .apply(pd.to_numeric, errors="coerce")
        )
        # Los valores 99, 999, etc. son datos válidos en varias variables del
        # Thies LPM (calidad, reflectividad, intensidad y conteos del espectro).
        # Por eso los centinelas solo se reemplazan cuando el usuario los indica
        # explícitamente y, de ser posible, restringidos a columnas conocidas.
        if error_sentinels is not None:
            sentinels = list(error_sentinels)
            target_columns = numeric_columns
            if sentinel_columns is not None:
                requested = set(sentinel_columns)
                target_columns = [col for col in numeric_columns if col in requested]
            if target_columns and sentinels:
                frame[target_columns] = frame[target_columns].mask(
                    frame[target_columns].isin(sentinels)
                )

    return frame


def _iter_filtered_chunks(
    file_path: Path,
    *,
    chunksize: Optional[int],
    precipitation_only: bool,
    precipitation_column: str,
    precipitation_threshold: float,
    quality_column: Optional[str],
    quality_threshold: float,
    drop_columns_after: Optional[Iterable[str]],
    error_sentinels: Optional[Iterable[float]],
    sentinel_columns: Optional[Iterable[str]],
    read_csv_kwargs: Dict[str, Any],
) -> Iterator[pd.DataFrame]:
    """Yield normalized (and optionally filtered) chunks from a CSV file."""
    if chunksize is not None and chunksize <= 0:
        raise ValueError("chunksize must be a positive integer or None")

    options = _build_read_csv_kwargs(read_csv_kwargs)
    if chunksize is not None:
        options["chunksize"] = chunksize

    reader = pd.read_csv(file_path, **options)
    if chunksize is None:
        reader = [reader]  # type: ignore[list-item]

    for chunk in reader:
        chunk = _normalize_disdro_frame(
            chunk,
            error_sentinels=error_sentinels,
            sentinel_columns=sentinel_columns,
        )
        if precipitation_only:
            if precipitation_column not in chunk.columns:
                raise KeyError(
                    f"Column '{precipitation_column}' not found in file '{file_path.name}'"
                )
            chunk = chunk[chunk[precipitation_column] > precipitation_threshold]
        if quality_column is not None:
            if quality_column not in chunk.columns:
                raise KeyError(
                    f"Column '{quality_column}' not found in file '{file_path.name}'"
                )
            chunk = chunk[chunk[quality_column] > quality_threshold]
        if chunk.empty:
            continue
        if drop_columns_after:
            chunk = chunk.drop(columns=list(drop_columns_after), errors="ignore")
        yield chunk


def iter_disdro_chunks(
    resultados_dir: Path = DEFAULT_RESULTS_DIR,
    station_id: Optional[str] = DEFAULT_STATION_ID,
    precipitation_only: bool = False,
    precipitation_column: str = PRECIPITATION_COLUMN,
    precipitation_threshold: float = 0.0,
    quality_column: Optional[str] = QUALITY_COLUMN,
    quality_threshold: float = QUALITY_THRESHOLD,
    chunksize: Optional[int] = DEFAULT_CHUNK_SIZE,
    error_sentinels: Optional[Iterable[float]] = None,
    sentinel_columns: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """Yield normalized and filtered ``(year, chunk)`` pairs.

    Esta es la interfaz de memoria acotada: cada bloque se entrega en cuanto se
    procesa. ``error_sentinels`` es optativo porque valores como 99 y 999 son
    válidos para varias variables del instrumento; usa ``sentinel_columns`` para
    limitar cualquier reemplazo a columnas documentadas.
    """
    for path in list_disdro_files(resultados_dir=resultados_dir, station_id=station_id):
        _, year = parse_station_year(path)
        read_kwargs = dict(read_csv_kwargs)
        required_columns: set[str] = set()
        if precipitation_only and precipitation_column:
            required_columns.add(precipitation_column)
        if quality_column:
            required_columns.add(quality_column)
        extra_columns: List[str] = []
        if required_columns:
            usecols = read_kwargs.get("usecols")
            if usecols is not None:
                if callable(usecols):
                    raise TypeError(
                        "usecols no puede ser una función cuando se aplican filtros; "
                        "pasa una secuencia de nombres de columna"
                    )
                usecols_list = list(usecols)
                usecols_set = set(usecols_list)
                missing = required_columns - usecols_set
                if missing:
                    ordered_missing = sorted(missing)
                    read_kwargs["usecols"] = usecols_list + ordered_missing
                    extra_columns = ordered_missing
            # if usecols is None, we read all columns so nothing extra to drop
        for chunk in _iter_filtered_chunks(
            path,
            chunksize=chunksize,
            precipitation_only=precipitation_only,
            precipitation_column=precipitation_column,
            precipitation_threshold=precipitation_threshold,
            quality_column=quality_column,
            quality_threshold=quality_threshold,
            drop_columns_after=extra_columns,
            error_sentinels=error_sentinels,
            sentinel_columns=sentinel_columns,
            read_csv_kwargs=read_kwargs,
        ):
            yield year, chunk.reset_index(drop=True)


def iter_disdro_files(
    resultados_dir: Path = DEFAULT_RESULTS_DIR,
    station_id: Optional[str] = DEFAULT_STATION_ID,
    precipitation_only: bool = False,
    precipitation_column: str = PRECIPITATION_COLUMN,
    precipitation_threshold: float = 0.0,
    quality_column: Optional[str] = QUALITY_COLUMN,
    quality_threshold: float = QUALITY_THRESHOLD,
    chunksize: Optional[int] = DEFAULT_CHUNK_SIZE,
    error_sentinels: Optional[Iterable[float]] = None,
    sentinel_columns: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> Iterator[Tuple[int, pd.DataFrame]]:
    """Yield one filtered DataFrame per year.

    Esta interfaz concatena los bloques de cada año. Para cálculos incrementales
    o equipos con poca RAM, usa :func:`iter_disdro_chunks`.
    """
    current_year: Optional[int] = None
    chunks: List[pd.DataFrame] = []

    for year, chunk in iter_disdro_chunks(
        resultados_dir=resultados_dir,
        station_id=station_id,
        precipitation_only=precipitation_only,
        precipitation_column=precipitation_column,
        precipitation_threshold=precipitation_threshold,
        quality_column=quality_column,
        quality_threshold=quality_threshold,
        chunksize=chunksize,
        error_sentinels=error_sentinels,
        sentinel_columns=sentinel_columns,
        **read_csv_kwargs,
    ):
        if current_year is not None and year != current_year:
            yield current_year, pd.concat(chunks, ignore_index=True)
            chunks = []
        current_year = year
        chunks.append(chunk)

    if current_year is not None and chunks:
        yield current_year, pd.concat(chunks, ignore_index=True)


def read_all_disdro_files(
    resultados_dir: Path = DEFAULT_RESULTS_DIR,
    station_id: Optional[str] = DEFAULT_STATION_ID,
    combine: bool = False,
    precipitation_only: bool = False,
    precipitation_column: str = PRECIPITATION_COLUMN,
    precipitation_threshold: float = 0.0,
    quality_column: Optional[str] = QUALITY_COLUMN,
    quality_threshold: float = QUALITY_THRESHOLD,
    chunksize: Optional[int] = DEFAULT_CHUNK_SIZE,
    error_sentinels: Optional[Iterable[float]] = None,
    sentinel_columns: Optional[Iterable[str]] = None,
    **read_csv_kwargs,
) -> Union[Dict[int, pd.DataFrame], pd.DataFrame]:
    """Read all available disdrometer CSV files.

    If ``combine`` is ``False`` (default), a dictionary mapping the year to each
    DataFrame is returned. When ``combine`` is ``True`` the individual frames are
    concatenated and a single DataFrame is returned. In the combined result a
    ``year`` column is injected so the source file can be identified.

    Use ``precipitation_only=True`` (por defecto sobre la columna ``var8``)
    para filtrar cada archivo a medida que se carga y conservar unicamente los
    eventos con precipitacion antes de concatenarlos. Adicionalmente se filtra
    por calidad utilizando ``quality_column`` (``var14``) y ``quality_threshold``
    (80.0 por defecto).

    El parametro ``chunksize`` controla cuantas filas se leen por bloque; el
    valor por defecto mantiene el consumo de memoria estable incluso con archivos
    muy grandes. Ajusta este numero segun tu capacidad de RAM.
    """
    combined_frames: List[pd.DataFrame] = []
    frames_by_year: Dict[int, pd.DataFrame] = {}

    for year, frame in iter_disdro_files(
        resultados_dir=resultados_dir,
        station_id=station_id,
        precipitation_only=precipitation_only,
        precipitation_column=precipitation_column,
        precipitation_threshold=precipitation_threshold,
        quality_column=quality_column,
        quality_threshold=quality_threshold,
        chunksize=chunksize,
        error_sentinels=error_sentinels,
        sentinel_columns=sentinel_columns,
        **read_csv_kwargs,
    ):
        if combine:
            if "year" not in frame.columns:
                frame = frame.copy()
                frame["year"] = year
            combined_frames.append(frame)
        else:
            frames_by_year[year] = frame

    if combine:
        if not combined_frames:
            return pd.DataFrame()
        return pd.concat(combined_frames, ignore_index=True)

    return frames_by_year


if __name__ == "__main__":
    if not DEFAULT_RESULTS_DIR.exists():
        print(
            f"No existe '{DEFAULT_RESULTS_DIR}'. Crea la carpeta y copia allí "
            "los archivos Disdrodata_417_YYYY.csv."
        )
    else:
        files = list_disdro_files()
        print(f"Encontrados {len(files)} archivos en '{DEFAULT_RESULTS_DIR}'.")
        for path in files:
            station, year = parse_station_year(path)
            print(f" - Estación {station} año {year}: {path.name}")
