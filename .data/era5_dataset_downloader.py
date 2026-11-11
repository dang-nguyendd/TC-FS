
import cdsapi
import os
import calendar
import time
from pathlib import Path

# ============================================================
# USER CONFIGURATION
# ============================================================

START_YEAR = 1979
END_YEAR = 1979

# Study area: North, West, South, East
AREA = [50, 100, 0, 180]

# Six-hourly observations (UTC)
TIMES = [
    "00:00",
    "06:00",
    "12:00",
    "18:00"
]

OUTPUT_DIR = Path("./ERA5_TC")
PRESSURE_DIR = OUTPUT_DIR / "pressure_levels"
LAND_DIR = OUTPUT_DIR / "land_invariants"

PRESSURE_DIR.mkdir(parents=True, exist_ok=True)
LAND_DIR.mkdir(parents=True, exist_ok=True)

client = cdsapi.Client()


# ============================================================
# GENERAL DOWNLOAD FUNCTION
# ============================================================

def download_file(dataset, request, filename):

    filename = Path(filename)

    # Skip completed downloads
    if filename.exists() and filename.stat().st_size > 0:
        print(f"SKIP: {filename}")
        return

    print(f"\nDownloading: {filename}")

    try:
        client.retrieve(
            dataset,
            request,
            str(filename)
        )

        if not filename.exists() or filename.stat().st_size == 0:
            raise RuntimeError("Downloaded file is missing or empty.")

        print(f"SUCCESS: {filename}")

    except Exception as error:

        # Remove incomplete files
        if filename.exists():
            filename.unlink()

        print(f"FAILED: {filename}")
        print(f"Error: {error}")

        # Continue with other requests
        return


# ============================================================
# DATASET 1: ERA5 PRESSURE LEVELS
# ============================================================

PRESSURE_DATASET = "reanalysis-era5-pressure-levels"

# Three groups reduce unnecessary variable/level combinations.
#
# Group 1:
# U and V wind at 200 and 850 hPa.
#
# Group 2:
# Temperature, relative vorticity and geopotential at
# 250, 500 and 850 hPa.
# Note: vorticity and geopotential at 250 and 500 hPa
# are additional fields, not required by your paper.
#
# Group 3:
# Specific humidity at 250, 500, 850 and 925 hPa.

PRESSURE_GROUPS = {

    "winds": {
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "pressure_level": ["200", "850"]
    },

    "temperature_vorticity_geopotential": {
        "variable": [
            "temperature",
            "vorticity",
            "geopotential"
        ],
        "pressure_level": ["250", "500", "850"]
    },

    "specific_humidity": {
        "variable": [
            "specific_humidity"
        ],
        "pressure_level": ["250", "500", "850", "925"]
    }
}


def download_era5_pressure_levels():

    print("\n======================================")
    print("DATASET 1: ERA5 PRESSURE LEVELS")
    print("======================================")

    for year in range(START_YEAR, END_YEAR + 1):

        for month in range(1, 13):

            month_str = f"{month:02d}"

            days = [
                f"{day:02d}"
                for day in range(
                    1,
                    calendar.monthrange(year, month)[1] + 1
                )
            ]

            for group_name, group in PRESSURE_GROUPS.items():

                filename = (
                    PRESSURE_DIR /
                    f"ERA5_{group_name}_{year}_{month_str}.nc"
                )

                request = {
                    "product_type": ["reanalysis"],

                    "variable": group["variable"],

                    "pressure_level": group["pressure_level"],

                    "year": [str(year)],
                    "month": [month_str],
                    "day": days,
                    "time": TIMES,

                    "area": AREA,

                    "data_format": "netcdf",
                    "download_format": "zip"
                }

                download_file(
                    PRESSURE_DATASET,
                    request,
                    filename
                )

                # Short pause between submissions
                time.sleep(2)


# ============================================================
# DATASET 2: ERA5-LAND INVARIANT VARIABLES
# ============================================================

LAND_DATASET = "reanalysis-era5-land"

LAND_VARIABLES = [

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


def download_era5_land_invariants():

    print("\n======================================")
    print("DATASET 2: ERA5-LAND INVARIANTS")
    print("======================================")

    filename = LAND_DIR / "ERA5_Land_Invariants.nc"

    # Invariant fields do not change with time.
    # A single reference date/time is used for the request.
    request = {

        "variable": LAND_VARIABLES,

        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],

        "area": AREA,

        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    download_file(
        LAND_DATASET,
        request,
        filename
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("ERA5 TC DATA DOWNLOAD")
    print(f"Period: {START_YEAR}-{END_YEAR}")
    print(f"Region: {AREA}")

    # Download atmospheric data
    download_era5_pressure_levels()

    # Download static land-surface data
    download_era5_land_invariants()

    print("\n======================================")
    print("DOWNLOAD PROCESS FINISHED")
    print("======================================")
