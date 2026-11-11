
import xarray as xr
import numpy as np
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================

input_file = (
    "./v04r01/access/netcdf/IBTrACS.WP.v04r01.nc"
)

output_file = (
    "./v04r01/IBTrACS.WP.filtered.nc"
)

START_DATE = "1979-01-01"
END_DATE = "2022-12-31 23:59:59"

MIN_WIND_MS = 17.2

# IBTrACS wind is normally stored in knots.
KNOT_TO_MS = 0.514444

# ==========================================
# READ DATA
# ==========================================

ds = xr.open_dataset(input_file, decode_times=True)

print(ds)
print("\nAvailable variables:")
print(list(ds.variables))

# ==========================================
# IDENTIFY VARIABLES
# ==========================================

wind_var = "wmo_wind"
time_var = "time" if "time" in ds else "date_time"

if wind_var not in ds:
    raise ValueError(
        f"{wind_var} not found. Check available wind variables."
    )

if time_var not in ds:
    raise ValueError("TC timestamp variable not found.")

storm_dim = ds["sid"].dims[0]

# ==========================================
# CONVERT WIND TO m/s
# ==========================================

wind = ds[wind_var]

units = wind.attrs.get("units", "knots").lower()

if "m/s" in units or "m s" in units:
    wind_ms = wind
else:
    # IBTrACS standard wind unit is knots.
    wind_ms = wind * KNOT_TO_MS

print("\nWind variable:", wind_var)
print("Original units:", units)

# Remove invalid/fill values.
wind_ms = wind_ms.where(
    (wind_ms >= 0) & (wind_ms < 150)
)

# ==========================================
# FILTER TIME PERIOD
# ==========================================

tc_time = ds[time_var]

valid_time = (
    (tc_time >= np.datetime64(START_DATE))
    & (tc_time <= np.datetime64(END_DATE))
)

# Mask observations outside the requested period.
wind_period = wind_ms.where(valid_time)

# ==========================================
# SELECT QUALIFYING STORMS
# ==========================================

# A storm qualifies if its maximum wind reaches
# at least 17.2 m/s during 1979-2022.

storm_max_wind = wind_period.max(
    dim=[d for d in wind_period.dims if d != storm_dim],
    skipna=True
)

qualifying_storms = (
    storm_max_wind >= MIN_WIND_MS
)

selected_indices = np.where(
    qualifying_storms.values
)[0]

print("\nTotal storms:", ds.sizes[storm_dim])
print("Qualifying storms:", len(selected_indices))

# ==========================================
# KEEP QUALIFYING STORMS
# ==========================================

filtered = ds.isel(storm=selected_indices)

# ==========================================
# FILTER DATE AND 6-HOURLY TIMESTAMPS
# ==========================================

# Use the actual datetime coordinate
tc_time = filtered["time"]

# Select date range
period_mask = (
    (tc_time >= np.datetime64("1979-01-01")) &
    (tc_time < np.datetime64("2023-01-01"))
)

# Select exact UTC hours: 00, 06, 12, 18
hour_mask = tc_time.dt.hour.isin([0, 6, 12, 18])

# Combine conditions
valid_mask = period_mask & hour_mask

# Select only time columns containing valid observations
keep_times = valid_mask.any(dim="storm")

time_indices = np.where(keep_times.values)[0]

filtered = filtered.isel(date_time=time_indices)

# Apply the observation-level mask
valid_mask = valid_mask.isel(date_time=time_indices)

# Mask invalid storm/time combinations
filtered = filtered.where(valid_mask)

# ==========================================
# VERIFY BEFORE SAVING
# ==========================================

print("\nFirst 20 timestamps:")

print(
    filtered["time"]
    .values[~np.isnat(filtered["time"].values)][:20]
)

print("\nUnique hours:")

print(
    np.unique(
        filtered["time"].dt.hour.values[
            ~np.isnat(filtered["time"].values)
        ]
    )
)

# ==========================================
# SAVE
# ==========================================

filtered.to_netcdf(
    output_file,
    encoding={
        var: {"zlib": True, "complevel": 4}
        for var in filtered.data_vars
        if filtered[var].dtype.kind in "fiu"
    }
)

print("\nFiltering completed.")
print("Output:", output_file)

ds.close()
filtered.close()