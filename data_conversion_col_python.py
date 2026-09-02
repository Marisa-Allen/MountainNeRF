"""
main code for rendering the depth maps for each camera. Have to first
read the colmap outputs and make into numpy arrays so that they
can be transfomred into the DEM coord system and then use
pyrender to get depth map for each camera pose
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rot
import trimesh

"""
Function that takes cameras.txt and converts in into numpy arrays
and a dicionary

inputs: path_to_camerastxt = path to file
outputs cameras = dictionary of camera values
"""
def cameras_parser(path_to_camerastxt):
    # creating dictionary 
    cameras = {}
    # opening file
    with open(path_to_camerastxt, "r") as f:
        # ignoring lines that start with # as comments
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    # itterating through lines and getting each value
    for line in lines:
        attri = line.split()
        camera_id = int(attri[0])
        model = attri[1]
        width = int(attri[2])
        height = int(attri[3])
        params = np.array(list(map(float, attri[4:])))
        
        # appending to dictionary
        cameras[camera_id] = {
            "model": model,
            "width": width,
            "height": height,
            "params": params,
            }
    return cameras
################


"""
Function that takes images.txt and converts in into numpy arrays
and a dicionary

inputs: path_to_imagestxt = path to file
outputs images = dictionary of imagevalues
"""
def images_parser(path_to_imagestxt):
    # creating empty dictionary
    images = {}
    # opening file
    with open(path_to_imagestxt, "r") as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    # alternate lines as info is spread across two lines
    for i in range(0, len(lines), 2):
        pose_parts = lines[i].split()
        image_id = int(pose_parts[0])
        qvec = np.array(list(map(float, pose_parts[1:5])))
        tvec = np.array(list(map(float, pose_parts[5:8])))
        camera_id = int(pose_parts[8])
        name = pose_parts[9]

        # extra values, not as useful but included
        other_parts = lines[i + 1].split()
        n_pts = len(other_parts) // 3
        xyz = np.zeros((n_pts, 2))
        point3D_ids = np.zeros(n_pts, dtype=np.int64)
        for j in range(n_pts):
            xyz[j, 0] = float(other_parts[j * 3])
            xyz[j, 1] = float(other_parts[j * 3 + 1])
            point3D_ids[j] = int(other_parts[j * 3 + 2])

        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
            # pixel point
            "xys": xyz,
            "point3D_ids": point3D_ids,
        }
    
    return images

###########

# function for taking the points3d.txt and converting to dict
def points3d_parser(path_to_points3d):
    points3d = {}
    with open(path_to_points3d, "r") as f:
        # not including lines with #
        lines = [l for l in f if not l.startswith("#") and l.strip()]
        
    for line in lines:
        parts = line.split()
        point_id = int(parts[0])
        
        # getting values
        xyz = np.array(list(map(float, parts[1:4])))
        rgb = np.array(list(map(int, parts[4:7])))
        error = float(parts[7])
        track_stuff = parts[8:]
        track = [
            (int(track_stuff[i]), int(track_stuff[i + 1]))
            for i in range(0, len(track_stuff), 2)
        ]
        points3d[point_id] = {
            "xyz": xyz,
            "rgb": rgb,
            "error": error,
            "track": track,
        }
        
        
        
    return points3d

# defining paths
cameras_path = "0_txt\cameras.txt"
images_path = "0_txt\images.txt"
points3d_path = "0_txt\points3D.txt"

# creating the dictionaries
cameras = cameras_parser(cameras_path)
images = images_parser(images_path)
d3points =  points3d_parser(points3d_path)

# transformation matrix - COLMAP to DEM
M = np.array([
    [-249.235115, -32.01858,  451.047011,  7345.544751],
    [-447.803045,  87.70212, -241.655595, -1679.918544],
    [-61.702081,  -508.26534,  -70.29852,  1886.38378],
    [0.000,  0.000,  0.000,     1.000]
])
R_sim = M[:3, :3]
scale = np.linalg.norm(R_sim, axis=1).mean()
R_only = R_sim / scale
t = M[:3, 3]

# functnion for transforming a point
def transform_point(p):
    return R_sim @ p + t

# function for transforming the camera pose
def transform_pose(qvec, tvec):
    # transformations
    R_cw = Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    C_old = -R_cw.T @ tvec
    C_new = transform_point(C_old)
    R_cw_new = R_cw @ R_only.T
    t_new = -R_cw_new @ C_new
    q_xyzw = Rot.from_matrix(R_cw_new).as_quat()
    q_new = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # back to (w,x,y,z)
    return q_new, t_new

# replacing the poses with transfomed ones
for image_id, img in images.items():
    img["qvec"], img["tvec"] = transform_pose(img["qvec"], img["tvec"])

# replacing the points with transformed ones
for point_id, pt in d3points.items():
    pt["xyz"] = transform_point(pt["xyz"])
    
    
# getting thinsg needed to render image
def get_camera_matrices(img, cameras):
    qvec, tvec = img["qvec"], img["tvec"]
    R_cw = Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    # camera centre
    C = -R_cw.T @ tvec

    # getting values
    cam = cameras[img["camera_id"]]
    model = cam["model"]
    w, h = cam["width"], cam["height"]
    p = cam["params"]
    
    # only using simple radial camera
    
    if model != "SIMPLE_RADIAL":
        raise ValueError(f"needed simple radial camera model instead got: {model}")

    # camera things
    f, cx, cy = p[0], p[1], p[2]
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

    return R_cw, C, K, w, h

import pyrender
# pyrender wants pose in world coordiantes

def render_dem_depth(mesh, C_dem, R_cw_dem, K, width, height):
    """
    mesh = mesh of DEM 
    C_dem = camera centre in DEM frame
    R_cw_dem = world to camera rotation in DEM frame
    K = intrinsics
    
    Returns depth map
    """
    # creating scene
    scene = pyrender.Scene()
    # mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(pyrender_mesh)

    # camera to world
    R_wc = R_cw_dem.T
    pose = np.eye(4)
    pose[:3, :3] = R_wc
    pose[:3, 3] = C_dem
    
    # flip y and z as pyrender uses something different - COLMAP and OpenGL
    cv_to_gl = np.diag([1, -1, -1, 1])
    GLpose = pose @ cv_to_gl
    # create virtual camera
    cam = pyrender.IntrinsicsCamera(
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
        znear=0.1, zfar=100000.0
    )
    scene.add(cam, pose=GLpose)

    r = pyrender.OffscreenRenderer(width, height)
    # camera space z depth not elucidian depth
    # depth is in meters
    depth = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    
    r.delete()
    return depth


import matplotlib.pyplot as plt

# code below is for testing on one image

mesh = trimesh.load("dem_mesh.ply")
# grab one image to test with
test_image_id = next(iter(images))
img = images[2000]

print("Testing image:", img["name"])
print(img["qvec"], img["tvec"])

R_cw, C, K, w, h = get_camera_matrices(img, cameras)

# renders as numpy arrays
depth = render_dem_depth(mesh, C, R_cw, K, w, h)

plt.imshow(depth, cmap="viridis")

# code for itterating through images and getting depth maps
# for each
"""
import os
from tqdm import tqdm

n_imgs = len(images)
out_dir = "dem_depths"
os.makedirs(out_dir, exist_ok=True)

for img_id, img in tqdm(images.items(), total=len(images)):
    R_cw, C, K, w, h = get_camera_matrices(img, cameras)
    depth = render_dem_depth(mesh, C, R_cw, K, w, h)
    base_name = os.path.splitext(img["name"])[0]
    npy_path = os.path.join(out_dir, f"{base_name}_depth.npy")
    np.save(npy_path, depth.astype(np.float32))
"""
