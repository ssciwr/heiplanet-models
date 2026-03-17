import numpy as np
import xarray as xr

# ---------------------------------------------------
# ------------------   pr dataset   -----------------
# ---------------------------------------------------
# Five longitudes
lon1 = np.array([-180.0, -179.5, -179.0, -178.5, -178.0], dtype=np.float64)
# Four latitudes
lat1 = np.array([-89.5, -89.0, -88.5, -88.0], dtype=np.float64)
# Two timestamps
time1 = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")

# shape (3, 4, 5) --> dimensions [time, latitude, longitude]
pr_data = np.array(
    [
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ],
        [
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40],
        ],
        [
            [41, 42, 43, 44, 45],
            [46, 47, 48, 49, 50],
            [51, 52, 53, 54, 55],
            [56, 57, 58, 59, 60],
        ],
    ],
    dtype=np.float64,
)

pr = xr.Dataset(
    data_vars={
        "tp": (["time", "latitude", "longitude"], pr_data),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
)

# Example [time, latitude, longitude]
pr_slice = pr.tp[0, :, :]
print(pr_slice.values)

# Write to netcdf file
pr.to_netcdf("data/in/Pratik_datalake/pr_dummy.nc")


# ---------------------------------------------------
# ----------------   dense dataset   ----------------
# ---------------------------------------------------

# Five longitudes
# lon2 = np.array([-179.75, -179.25, -178.75, -178.25, -177.75])
lon2 = np.array([-180.0, -179.5, -179.0, -178.5, -178.0], dtype=np.float64)

# Four latitudes
# lat2 = np.array([-89.75, -89.25, -88.75, -88.25])
lat2 = np.array([-89.5, -89.0, -88.5, -88.0], dtype=np.float64)

# One timestamp
time2 = np.array(["2024-01-01"], dtype="datetime64[ns]")
dens_data = np.array(
    [
        [
            [-1, -2, -3, -4, -5],
            [-6, -7, -8, -9, -10],
            [-11, -12, -13, -14, -15],
            [-16, -17, -18, -19, -20],
        ],
    ],
    dtype=np.float64,
)

dens = xr.Dataset(
    data_vars={
        "dens": (["time", "lat", "lon"], dens_data),
    },
    coords={
        "lon": lon2,
        "lat": lat2,
        "time": time2,
    },
)

# Example
dens_slice = dens.dens[0, :, :]
print(dens_slice.values)


# Write to netcdf file
dens.to_netcdf("data/in/Pratik_datalake/dense_dummy.nc")


# Example for operations
# result = pr_slice + dens_slice

# print(result.compute())

# Interpolate dens to pr's grid
# print("Result aligned")
# dens_aligned = dens.interp(
#    longitude=pr.longitude,
#    latitude=pr.latitude,
#    method="linear",  # or "linear" for smoother interpolation
# )

# ------------------------------------------------------------
# ------------------   temperature dataset   -----------------
# ------------------------------------------------------------
# shape (3, 4, 5) --> dimensions [time, latitude, longitude]
temperature_data = np.array(
    [
        [
            [2, 4, 6, 8, 10],
            [12, 14, 16, 18, 20],
            [22, 24, 26, 28, 30],
            [32, 34, 36, 38, 40],
        ],
        [
            [-2, -4, -6, -8, -10],
            [-12, -14, -16, -18, -20],
            [-22, -24, -26, -28, -30],
            [-32, -34, -36, -38, -40],
        ],
        [
            [1.5, 2, 2.5, 3, 3.5],
            [4, 4.5, 5, 5.5, 6],
            [6.5, 7, 7.5, 8, 8.5],
            [9, 9.5, 10, 10.5, 11],
        ],
    ],
    dtype=np.float64,
)

temperature = xr.Dataset(
    data_vars={
        "t2m": (["time", "latitude", "longitude"], temperature_data),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
)

# Example [time, latitude, longitude]
temperature_slice = temperature.t2m[0, :, :]
print(temperature_slice.values)

# Write to netcdf file
temperature.to_netcdf("data/in/Pratik_datalake/temperature_dummy.nc")
