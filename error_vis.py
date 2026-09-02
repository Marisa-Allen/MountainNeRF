"""
This code takes the COLMAP sparse point cloud and the DEM point cloud
transforms the COLMAP cloud so it lines up with the DEM. Then for each
DEM point, finds the closest COLMAP point in XY plane and compares thier
elevation. the difference is the elevation erorrs
"""
# Imporing libaries
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# reading point clouds
dem = o3d.io.read_point_cloud("point_cloud_cropp.ply")
col = o3d.io.read_point_cloud("chile_ply.ply")


# transformation matrix between colmap sparse and DEM - final transformation
M = np.array([
    [-249.235115, -32.01858,  451.047011,  7345.544751],
    [-447.803045,  87.70212, -241.655595, -1679.918544],
    [-61.702081,  -508.26534,  -70.29852,  1886.38378],
    [0.000,  0.000,  0.000,     1.000]
])
# transforming point cloud
col.transform(M)


# Making into arrays - numpy shape (M,3) with each row (X, Y, Z)
dem_pts = np.asarray(dem.points)
col_pts = np.asarray(col.points)

# building a KD tree, only using X and Y
tree = cKDTree(col_pts[:, :2])
# maximum matching radius
radius = 30.0
# Finding nearest COLMAP point
dist_xy, idx = tree.query(
    dem_pts[:, :2],
    distance_upper_bound=radius
)

idx == len(col_pts)
valid = idx < len(col_pts)

# filtering- keep only matched points
dem_valid = dem_pts[valid]
col_valid = col_pts[idx[valid]]
# Elevation error
dz = dem_valid[:,2] - col_valid[:,2]

# How many matched points
print(f"Matched points : {len(dz)}")

# if there is an offset - yes
print(f"Mean bias      : {dz.mean():.3f}")
# mean absolute error - average magnitude of the elevation error
print(f"MAE            : {np.mean(np.abs(dz)):.3f}")
# Root Mean Square Error - typical error with 
# greater sensitivity to outliers
print(f"RMSE           : {np.sqrt(np.mean(dz**2)):.3f}")
# standard deviation
print(f"Std dev        : {dz.std():.3f}")
# used for visulastion
print(f"95th percentile: {np.percentile(np.abs(dz),95):.3f}")

plt.figure(figsize=(12, 8))

# Use the 95th percentile as the colour scale limit - looks better
max_error = np.percentile(np.abs(dz), 95)

plt.scatter(
    dem_valid[:, 0],
    dem_valid[:, 1],
    c=dz,
    cmap="coolwarm",
    vmin=-max_error,
    vmax=max_error,
    s=2
)

plt.colorbar(label="Elevation Error (m)")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.title("COLMAP Sparse Cloud and DEM Vertical Error")

plt.axis("equal")
plt.savefig(
    "elevation_error_map.pdf",
    format="pdf",
    bbox_inches="tight"
)
plt.show()

error_cloud = o3d.geometry.PointCloud()
error_cloud.points = o3d.utility.Vector3dVector(dem_valid)

# Normalize errors for coloring
max_error = np.percentile(np.abs(dz), 95)

norm = np.clip((dz + max_error) / (2 * max_error), 0, 1)

# Blue = negative error
# White = zero
# Red = positive error
colors = plt.cm.coolwarm(norm)[:, :3]

error_cloud.colors = o3d.utility.Vector3dVector(colors)

#o3d.visualization.draw_geometries(
#    [error_cloud],
#    window_name="DEM elevation error (dz)"
#)

