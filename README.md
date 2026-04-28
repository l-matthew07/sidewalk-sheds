# SidewalkShed

Minimal Python scraper for NYC sidewalk shed / scaffold permit locations.

It does three things:

1. Probes the live NYC Open Data endpoints and prints sample raw fields.
2. Builds a cleaned list of active scaffold permit locations.
3. Fills in missing coordinates with NYC GeoClient or Nominatim, then writes `scaffolds.json`.
4. Serves a one-page Flask + Leaflet viewer for route inspection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 fetch_scaffolds.py
```

## Run The Web App

```bash
python3 app.py
```

Then open `http://127.0.0.1:5000` and enter two addresses to compare routes.

## Optional environment variables

- `SOCRATA_APP_TOKEN`: optional app token for NYC Open Data.
- `NYC_GEOCLIENT_APP_ID`: use NYC GeoClient for missing coordinates.
- `NYC_GEOCLIENT_APP_KEY`: paired with `NYC_GEOCLIENT_APP_ID`.
- `NOMINATIM_EMAIL`: optional contact email for Nominatim requests.
- `GOOGLE_MAPS_API_KEY`: optional. If set, `/api/geocode` uses Google Geocoding first and falls back to Nominatim if needed.

If GeoClient credentials are not set, the script falls back to Nominatim.
