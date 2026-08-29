#!/usr/bin/env python
"""Descarga las fuentes oficiales SIATA/NOAA y extrae la estación 417.

La descarga se reanuda automáticamente después de cortes de conexión. El
manifiesto final registra URL, tamaño, fecha HTTP y SHA-256 de cada fuente.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = BASE_DIR / "datos_fuente"
RESULTS_DIR = BASE_DIR / "resultados"
MANIFEST_PATH = SOURCE_DIR / "manifest.json"

SOURCES = {
    "siata_metadata": {
        "url": "https://siata.gov.co/usuarios/jsepulveda/disdrometro/Metadata_Disdros.csv",
        "path": SOURCE_DIR / "siata" / "Metadata_Disdros.csv",
    },
    "siata_variables": {
        "url": "https://siata.gov.co/usuarios/jsepulveda/disdrometro/varsDisdroBD_names.xlsx",
        "path": SOURCE_DIR / "siata" / "varsDisdroBD_names.xlsx",
    },
    "siata_historico": {
        "url": "https://siata.gov.co/usuarios/jsepulveda/disdrometro/historico_disdro.tar.gz",
        "path": SOURCE_DIR / "siata" / "historico_disdro.tar.gz",
    },
    "noaa_omega": {
        "url": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/pressure/omega.mon.mean.nc",
        "path": SOURCE_DIR / "noaa" / "omega.mon.mean.nc",
    },
    "noaa_chi": {
        "url": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/sigma/chi.mon.mean.nc",
        "path": SOURCE_DIR / "noaa" / "chi.mon.mean.nc",
    },
}


def request(url: str, *, method: str = "GET", start: int | None = None):
    headers = {"User-Agent": "T2-Disdrometro-reproducibilidad/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    return Request(url, headers=headers, method=method)


def remote_metadata(url: str) -> dict[str, str | int | None]:
    with urlopen(request(url, method="HEAD"), timeout=90) as response:
        length = response.headers.get("Content-Length")
        return {
            "bytes": int(length) if length else None,
            "last_modified": response.headers.get("Last-Modified"),
            "content_type": response.headers.get("Content-Type"),
        }


def download_resumable(url: str, destination: Path, expected_bytes: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 101):
        current = destination.stat().st_size if destination.exists() else 0
        if expected_bytes is not None and current == expected_bytes:
            return
        if expected_bytes is not None and current > expected_bytes:
            raise RuntimeError(f"{destination} supera el tamaño remoto esperado")

        try:
            with urlopen(request(url, start=current), timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if current and status != 206:
                    raise RuntimeError(
                        f"El servidor no aceptó reanudar {destination.name} (HTTP {status})"
                    )
                mode = "ab" if current else "wb"
                with destination.open(mode) as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            print(f"Corte en {destination.name} (intento {attempt}): {exc}")
            time.sleep(min(5 * attempt, 60))
            continue

        new_size = destination.stat().st_size
        if expected_bytes is None or new_size == expected_bytes:
            return
        print(f"Reanudando {destination.name}: {new_size}/{expected_bytes} bytes")

    raise RuntimeError(f"No fue posible completar {destination.name}")


def extract_station_417(archive_path: Path) -> list[Path]:
    """Extract only annual station-417 CSV files, never other stations."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^Disdrodata_417_(\d{4})\.csv$", re.IGNORECASE)
    extracted: list[Path] = []

    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            basename = Path(member.name).name
            if not member.isfile() or not pattern.match(basename):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            destination = RESULTS_DIR / basename
            temporary = destination.with_suffix(destination.suffix + ".part")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            temporary.replace(destination)
            extracted.append(destination)
            print(f"Extraído: {destination.name} ({destination.stat().st_size} bytes)")

    if not extracted:
        raise RuntimeError("El archivo SIATA no contiene CSV reconocibles de la estación 417")
    return sorted(extracted)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(http_metadata: dict[str, dict], extracted: list[Path]) -> None:
    files = {}
    for name, source in SOURCES.items():
        path = source["path"]
        files[name] = {
            "url": source["url"],
            "relative_path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            **http_metadata[name],
        }
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "station": {
            "id": 417,
            "name": "Santa Elena Radar - Disdrometro",
            "latitude": 6.1935,
            "longitude": -75.5276,
        },
        "sources": files,
        "extracted_station_files": [
            {
                "relative_path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in extracted
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    metadata: dict[str, dict] = {}
    for name, source in SOURCES.items():
        print(f"Verificando {name}: {source['url']}")
        metadata[name] = remote_metadata(source["url"])
        if not args.skip_download:
            download_resumable(source["url"], source["path"], metadata[name]["bytes"])

    if args.skip_extract:
        extracted = sorted(RESULTS_DIR.glob("Disdrodata_417_*.csv"))
    else:
        extracted = extract_station_417(SOURCES["siata_historico"]["path"])
    write_manifest(metadata, extracted)
    print(f"Manifiesto reproducible: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
