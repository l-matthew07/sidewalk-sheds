# SidewalkShed

Minimal Python scraper for NYC sidewalk shed / scaffold permit locations.

It does three things:

1. Probes the live NYC Open Data endpoints and prints sample raw fields.
2. Builds a cleaned list of active scaffold permit locations.
3. Fills in missing coordinates with NYC GeoClient or Nominatim, then writes `scaffolds.json`.
4. Precomputes a lightweight Manhattan routing bundle for deployment.
5. Serves a one-page Flask + Leaflet viewer for route inspection.

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

## Precompute Routing Data

The production app does not use `osmnx` or `networkx` at runtime. Instead, generate a lightweight routing bundle locally:

```bash
python3 precompute.py
```

This creates:

- `graph_data.pkl`: committed deployment artifact used by `app.py`
- `scaffold_edges.json`: optional debug output showing scaffold counts per edge

If you need to run `precompute.py`, install `osmnx` and `networkx` locally in your own environment. They are intentionally not listed in `requirements.txt` because they should never be installed on the server.

## Run The Web App

```bash
python3 app.py
```

Then open `http://127.0.0.1:5000` and enter two addresses to compare routes.

## Deploy On Render

Use this start command:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

Make sure these files are present in the deployed repo:

- `scaffolds.json`
- `graph_data.pkl`

## Optional environment variables

- `SOCRATA_APP_TOKEN`: optional app token for NYC Open Data.
- `NYC_GEOCLIENT_APP_ID`: use NYC GeoClient for missing coordinates.
- `NYC_GEOCLIENT_APP_KEY`: paired with `NYC_GEOCLIENT_APP_ID`.
- `NOMINATIM_EMAIL`: optional contact email for Nominatim requests.
- `GOOGLE_MAPS_API_KEY`: optional. If set, `/api/geocode` uses Google Geocoding first and falls back to Nominatim if needed.

If GeoClient credentials are not set, the script falls back to Nominatim.
