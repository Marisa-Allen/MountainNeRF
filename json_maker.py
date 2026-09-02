"""
Code for making json files for nerfacto from COLMAP. Takes the txt
version of the COLMAP output. Transformed also to the DEM frame
from COLMAP frame
"""
# importing libaries
import pycolmap
import numpy as np
import json
from pathlib import Path

# Getting the COLMAP model
reconstruction = pycolmap.Reconstruction("0_txt")

frames = []

image = next(iter(reconstruction.images.values()))

# looping through frames + transfomring  saving
for image_id, image in reconstruction.images.items():
    camera = reconstruction.cameras[image.camera_id]
    
    # Transformation matrix COLMAP -> DEM
    T = np.array([
        [-249.235115, -32.01858,  451.047011,  7345.544751],
        [-447.803045,  87.70212, -241.655595, -1679.918544],
        [-61.702081,  -508.26534,  -70.29852,  1886.38378],
        [0.000,  0.000,  0.000,     1.000]
    ])
    
    
    # Rotation part
    Rot = T[:3, :3]
    # translation part
    trans = T[:3, 3]
    scale = np.linalg.norm(Rot, axis=1).mean()
    R_only = Rot/scale
    
    # Function to transform a point
    def transform_point(p):
        return Rot @ p + trans
    
    # To open GL format
    cv_to_gl = np.diag([1, -1, -1, 1])
    
    # Pose of the camera
    pose_matrix = image.cam_from_world().matrix()
    # R and t camera to world
    R_cw = pose_matrix[:, :3]
    t_cw = pose_matrix[:, 3]

    # transforming
    C_old = -R_cw.T @ t_cw
    C_new = transform_point(C_old)
    R_cw_new = R_cw @ R_only.T
    # world to camera
    R_wc_new = R_cw_new.T
    
    T_wc_dem = np.eye(4)
    T_wc_dem[:3, :3] = R_wc_new
    T_wc_dem[:3, 3] = C_new
    # going to open GL format
    T_wc_dem = T_wc_dem @ cv_to_gl
    
    # creating and adding info for frame
    frame = {
        "file_path": f"images/{image.name}",
        "depth_file_path": f"dem_depths/{Path(image.name).stem}_depth.npy",
        "transform_matrix": T_wc_dem.tolist(),
    }

    frames.append(frame)

cam = next(iter(reconstruction.cameras.values()))

# creating the json file
transforms = {
    "w": cam.width,
    "h": cam.height,
    "fl_x": cam.focal_length_x,
    "fl_y": cam.focal_length_y,
    "cx": cam.principal_point_x,
    "cy": cam.principal_point_y,
    "camera_model": "OPENCV",
    "frames": frames,
}

# saving
with open("transforms.json", "w") as f:
    json.dump(transforms, f, indent=2)
