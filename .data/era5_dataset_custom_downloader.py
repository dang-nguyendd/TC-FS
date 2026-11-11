
import cdsapi
import xarray as xr
import pandas as pd
import numpy as np

from pathlib import Path
import calendar
import time
import math


# ============================================================
# 1. CONFIGURATION
# ============================================================

IBTRACS_FILE = (
    r"./.data/v04r01/access/netcdf/IBTrACS.WP.v04r01.nc"
)

OUTPUT_DIR = Path("./ERA5_TC_IBTRACS")

PRESSURE_DIR = OUTPUT_DIR / "pressure_levels"
LAND_DIR = OUTPUT_DIR / "land_invariants"

PRESSURE_DIR.mkdir(parents=True, exist_ok=True)
LAND_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1979
END_YEAR = 2022

# Spatial buffer around TC observations (degrees)
SPATIAL_BUFFER = 5.0

# Western North Pacific limits
BASIN_LAT_MIN = 0
BASIN_LAT_MAX = 50
BASIN_LON_MIN = 100
BASIN_LON_MAX = 180

# ERA5 grid resolution
GRID_RESOLUTION = 0.25

# Download format
DATA_FORMAT = "netcdf"

client = cdsapi.Client()


# ============================================================
# 2. READ IBTRACS NETCDF
# ============================================================

def find_variable(ds, candidates):

    for name in candidates:
        if name in ds.variables:
            return name

    raise ValueError(
        f"Cannot find any of these variables: {candidates}\n"
        f"Available variables: {list(ds.variables)}"
    )


def load_ibtracs():

    print("Reading IBTrACS dataset...")

    ds = xr.open_dataset(
        IBTRACS_FILE,
        decode_times=True
    )

    print("\nIBTrACS dimensions:")
    print(ds.dims)

    print("\nIBTrACS variables:")
    print(list(ds.variables))

    # Variable names in IBTrACS NetCDF can differ from
    # names used in CSV or other distributions.
    time_var = find_variable(
        ds, ["time", "iso_time", "ISO_TIME"]
    )

    lat_var = find_variable(
        ds, ["lat", "latitude", "LAT"]
    )

    lon_var = find_variable(
        ds, ["lon", "longitude", "LON"]
    )

    # Convert variables to a tabular structure.
    tracks = xr.Dataset({
        "time": ds[time_var],
        "lat": ds[lat_var],
        "lon": ds[lon_var]
    }).to_dataframe().reset_index()

    ds.close()

    # Remove missing observations.
    tracks = tracks.dropna(
        subset=["time", "lat", "lon"]
    ).copy()

    tracks["time"] = pd.to_datetime(
        tracks["time"],
        errors="coerce",
        utc=True
    )

    tracks = tracks.dropna(subset=["time"])

    tracks["lat"] = pd.to_numeric(
        tracks["lat"], errors="coerce"
    )

    tracks["lon"] = pd.to_numeric(
        tracks["lon"], errors="coerce"
    )

    tracks = tracks.dropna(
        subset=["lat", "lon"]
    )

    # Remove invalid coordinates and fill values.
    tracks = tracks[
        tracks["lat"].between(-90, 90)
        & tracks["lon"].between(-180, 360)
    ].copy()

    # Convert longitude to 0-360 convention.
    tracks["lon"] = tracks["lon"] % 360

    # Keep the requested period.
    # tracks = tracks[
    #     tracks["time"].dt.year.between(
    #         START_YEAR, END_YEAR
    #     )
    # ].copy()

    # Restrict to the western North Pacific study area.
    tracks = tracks[
        tracks["lat"].between(
            BASIN_LAT_MIN, BASIN_LAT_MAX
        )
        & tracks["lon"].between(
            BASIN_LON_MIN, BASIN_LON_MAX
        )
    ].copy()

    # ERA5 is hourly. Round IBTrACS timestamps to
    # the nearest hour.
    tracks["era5_time"] = (
        tracks["time"].dt.round("h")
    )

    # Save the matched timestamps for later verification.
    tracks.to_csv(
        OUTPUT_DIR / "IBTrACS_ERA5_timestamp_mapping.csv",
        index=False
    )

    print("\nIBTrACS observations:")
    print(len(tracks))

    print("\nUnique ERA5 timestamps:")
    print(tracks["era5_time"].nunique())

    print("\nDate range:")
    print(tracks["era5_time"].min())
    print(tracks["era5_time"].max())

    return tracks


# ============================================================
# 3. DEFINE ERA5 VARIABLE GROUPS
# ============================================================

# Important: ERA5 CDS variable identifier for
# relative vorticity is "vorticity".

PRESSURE_GROUPS = {

    "wind": {
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "pressure_level": ["200", "850"]
    },

    "temperature": {
        "variable": ["temperature"],
        "pressure_level": ["250", "500", "850"]
    },

    "vorticity": {
        "variable": ["vorticity"],
        "pressure_level": ["850"]
    },

    "specific_humidity": {
        "variable": ["specific_humidity"],
        "pressure_level": ["250", "500", "850", "925"]
    },

    "geopotential": {
        "variable": ["geopotential"],
        "pressure_level": ["850"]
    }
}

# ============================================================
# 4. CREATE A DAILY SPATIAL BOUNDING BOX
# ============================================================

def get_daily_area(day_tracks):

    lat_min = day_tracks["lat"].min()
    lat_max = day_tracks["lat"].max()

    lon_min = day_tracks["lon"].min()
    lon_max = day_tracks["lon"].max()

    north = min(
        BASIN_LAT_MAX,
        lat_max + SPATIAL_BUFFER
    )

    south = max(
        BASIN_LAT_MIN,
        lat_min - SPATIAL_BUFFER
    )

    west = max(
        BASIN_LON_MIN,
        lon_min - SPATIAL_BUFFER
    )

    east = min(
        BASIN_LON_MAX,
        lon_max + SPATIAL_BUFFER
    )

    # Round outward to ERA5 grid boundaries.
    north = math.ceil(north / GRID_RESOLUTION) * GRID_RESOLUTION
    south = math.floor(south / GRID_RESOLUTION) * GRID_RESOLUTION
    west = math.floor(west / GRID_RESOLUTION) * GRID_RESOLUTION
    east = math.ceil(east / GRID_RESOLUTION) * GRID_RESOLUTION

    # CDS order: North, West, South, East.
    return [
        round(north, 2),
        round(west, 2),
        round(south, 2),
        round(east, 2)
    ]


# ============================================================
# 5. DOWNLOAD ONE REQUEST
# ============================================================

def download_request(dataset, request, filename):

    filename = Path(filename)

    if filename.exists() and filename.stat().st_size > 0:

        print(f"SKIP: {filename}")
        return True

    print(f"\nDownloading: {filename}")

    for attempt in range(1, 4):

        try:

            client.retrieve(
                dataset,
                request,
                str(filename)
            )

            if not filename.exists() or filename.stat().st_size == 0:
                raise RuntimeError("Empty or missing download.")

            print(f"SUCCESS: {filename}")
            return True

        except Exception as error:

            print(
                f"Attempt {attempt}/3 failed: {error}"
            )

            if filename.exists():
                filename.unlink()

            if attempt < 3:
                time.sleep(30 * attempt)

    print(f"FAILED: {filename}")

    return False


# ============================================================
# 6. DOWNLOAD ERA5 FOR TC OBSERVATION DATES
# ============================================================

def download_era5_pressure(tracks):

    dataset = "reanalysis-era5-pressure-levels"

    # Group by calendar day.
    tracks = tracks.copy()

    tracks["date"] = (
        tracks["era5_time"].dt.strftime("%Y-%m-%d")
    )

    daily_groups = tracks.groupby("date")

    print("\nNumber of TC observation days:")
    print(len(daily_groups))

    for date_string, day_tracks in daily_groups:

        date = pd.Timestamp(date_string)

        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        # Unique hours observed on this day.
        hours = sorted(
            day_tracks["era5_time"]
            .dt.strftime("%H:00")
            .unique()
            .tolist()
        )

        # Daily TC bounding box + buffer.
        area = get_daily_area(day_tracks)

        print("\n--------------------------------")
        print(f"Date: {date_string}")
        print(f"Times: {hours}")
        print(f"Area: {area}")
        print(f"TC observations: {len(day_tracks)}")

        for group_name, group in PRESSURE_GROUPS.items():

            filename = (
                PRESSURE_DIR
                / year
                / month
                / f"ERA5_{date_string}_{group_name}.nc"
            )

            filename.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            request = {

                "product_type": ["reanalysis"],

                "variable": group["variable"],

                "pressure_level": group["pressure_level"],

                "year": [year],
                "month": [month],
                "day": [day],

                "time": hours,

                "area": area,

                "data_format": DATA_FORMAT,
                "download_format": "unarchived"
            }

            download_request(
                dataset,
                request,
                filename
            )

            time.sleep(2)


# ============================================================
# 7. DOWNLOAD ERA5-LAND INVARIANT VARIABLES
# ============================================================

def download_era5_land_invariants():

    dataset = "reanalysis-era5-land"

    filename = (
        LAND_DIR / "ERA5_Land_Invariants.nc"
    )

    variables = [

        "high_vegetation_cover",
        "low_vegetation_cover",

        "lake_cover",
        "lake_depth",

        "land_sea_mask",

        "soil_type",

        "type_of_high_vegetation",
        "type_of_low_vegetation",

        "geopotential"
    ]

    request = {

        "variable": variables,

        # Invariant fields: one reference date is sufficient.
        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],

        "area": [
            BASIN_LAT_MAX,
            BASIN_LON_MIN,
            BASIN_LAT_MIN,
            BASIN_LON_MAX
        ],

        "data_format": DATA_FORMAT,
        "download_format": "unarchived"
    }

    download_request(
        dataset,
        request,
        filename
    )


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    tracks = load_ibtracs()

    if tracks.empty:
        raise RuntimeError(
            "No valid IBTrACS observations found."
        )

    # Download only atmospheric data corresponding
    # to IBTrACS observation dates and hours.
    download_era5_pressure(tracks)

    # Static land-surface variables are downloaded once.
    download_era5_land_invariants()

    print("\n========================================")
    print("ALL DOWNLOAD REQUESTS FINISHED")
    print("========================================")
