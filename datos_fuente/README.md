# Fuentes de datos

Los datos crudos no se almacenan en GitHub. El análisis utiliza exclusivamente
las siguientes fuentes públicas:

## SIATA

Directorio oficial:
https://siata.gov.co/usuarios/jsepulveda/disdrometro/

- `historico_disdro.tar.gz` — histórico completo; el script extrae únicamente
  `Disdrodata_417_2019.csv` a `Disdrodata_417_2025.csv`.
- `Metadata_Disdros.csv` — identifica la estación 417 en Santa Elena.
- `varsDisdroBD_names.xlsx` — metadatos publicados de variables.

## NOAA PSL

Catálogo oficial:
https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html

- `Monthlies/pressure/omega.mon.mean.nc` — OMEGA mensual, nivel 500 hPa.
- `Monthlies/sigma/chi.mon.mean.nc` — potencial de velocidad CHI mensual,
  nivel sigma 0.2101.

Los archivos NOAA se actualizan con el tiempo. `manifest.json` conserva URL,
tamaño, fecha HTTP y SHA-256 de la instantánea que produjo los resultados
publicados. El histórico SIATA descargado ocupó 1.602.246.060 bytes; las cinco
fuentes sumaron 2.079.762.204 bytes y los siete CSV extraídos, 5.276.486.900.

## Descarga

Desde la raíz del repositorio:

```powershell
.\.venv\Scripts\python.exe scripts\descargar_datos.py
```

La descarga se reanuda automáticamente después de un corte y no extrae datos de
otras estaciones.

