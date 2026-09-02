"""
This code visualises the COLMAP sparse point cloud and the DEM
point cloud to see how sucsessful the alignemnt was visually.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

pcd_dem = o3d.io.read_point_cloud("point_cloud_cropp.ply")
pcd_col = o3d.io.read_point_cloud("chile_ply.ply")


points = np.load("dem_test.npy")

x_dem = points[:, 0]
y_dem = points[:, 1]

lon0 = x_dem.mean()
lat0 = y_dem.mean()
print(f"lon0 = {lon0}")
print(f"lat0 = {lat0}")

# transformation after point pairs to get initial match
T = np.array([
    [0.729, -0.281,  0.043,  7960.815],
    [0.284,  0.720, -0.115, -1266.662],
    [0.002,  0.123,  0.773,  1704.135],
    [0.000,  0.000,  0.000,     1.000]
])

# transformation after icp to refine
T2 = np.array([
    [0.996, 0.085,  0.020,  -126.561],
    [-0.085,  0.996, -0.019, 596.918],
    [-0.022,  0.017,  1.00,  202.722],
    [0.000,  0.000,  0.000,     1.000]
])
# bwlow was done intially to get the first match/ get them near each other
"""
# inital orientation
R = pcd_col.get_rotation_matrix_from_xyz((np.pi/2, np.pi/4, np.pi))

pcd_col.rotate(R, center=([-0.83261547, 0.39001603, 0.41736928]))

pcd_col.translate([0.83261547, -0.39001603, -0.41736928])

pcd_col.scale(660, center=[5.81120028e-14, -1.15315571e-14, -1.25178953e-11])

pcd_col.transform(T)
pcd_col.transform(T2)
"""
# full transformation matrix that takes the COLMAP point cloud and
# Transforms it to the DEM world frame
T_total = np.array([
    [ -249.235,  -32.01858,  451.047,  7345.5448 ],
    [-447.803,  87.70212,  -241.6556, -1679.919 ],
    [  -61.7021, -508.265,  -70.2985,  1886.3838 ],
    [   0.000,    0.000,    0.000,     1.000 ]
])

pcd_col.transform(T_total)

#pcd_dem.scale(0.5, center=pcd_dem.get_center())
#o3d.io.write_point_cloud("col_transf_preicp.ply", pcd_col)

#o3d.visualization.draw_geometries([pcd_dem, pcd_col])

o3d.visualization.draw_geometries([pcd_dem])

