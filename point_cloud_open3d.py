"""
This code opens and reads the saved numpy version of the DEM
and turns it into an open 3D point cloud. This needs to
be trasformed so it can be viewed and worked with. Therefore it is
trasfomred to a local ENU like Cartesian coordinate system centered on
the mean lat and longditude of the DEM using equirectangular aproximation
which is Plate Carrée approximation. Distance will now be in meters.
"""
# importing libaries
import open3d as o3d
import numpy as np

# loading the DEM saved as a numpy file
dem = np.load("dem_test.npy")

# x y and z, loading and seperating
x_dem = dem[:, 0]
y_dem = dem[:, 1]
elev = dem[:, 2]


y_dem0 = y_dem.mean()
# 1 deg of lattitude = 111,320 m
meters_deg_lati = 111320
meters_deg_long = 111320 * np.cos(np.radians(y_dem0))

# converting
x_new = (x_dem - x_dem.mean()) * meters_deg_long
y_new = (y_dem - y_dem.mean()) * meters_deg_lati
z_new = elev
#print(x_dem.mean(), y_dem.mean())

# Creating the new point cloud
points = np.column_stack((x_new, y_new, z_new))

# making into open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Saving 
#o3d.io.write_point_cloud("point_cloud.ply", pcd)

