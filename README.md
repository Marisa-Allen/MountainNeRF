# MountainNeRF

Code for MountainNeRF, point cloud alignment and video results.
Within MountainNeRF/mountainnerf:
dem_feild.py - code for creating a field from the DEM that can  be quieried
dem_height_config.py - configuration code for the model
dem_height_model.py - Code for MountainNeRf with height loss and floater supression loss

Other code:
point_cloud_open3d.py - code to convert the .tif DEM into an open3d point cloud
point_cloud_vis.py - code for visualising point clouds
error_vis.py - code for viewing error between point clouds
dem_mesh_maker.py - code for making a mesh from the DEM with Trimesh
data_conversion_col_python.py - code for converting COLMAP data to numpy arrays and rendering depth maps for each camera pose
json_maker.py - code for making json files

Results:
Nerfacto_Render.mp4 - video render from model of plain Nerfacto
DepthNerfacto_Render.mp4 - Depth nerfacto video
MountainNeRF_base_DepthNerfacto.mp4 - render of MountainNeRF with depth-nerfacto
MountainNeRF_base_Nerfacto.mp4 - render of MountainNeRF with nerfacto
