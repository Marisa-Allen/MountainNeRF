"""
This code converts the DEM to a mesh to that it becomes a surface
This is so that the depth for each camera pose can be rendered.
"""
# Importing libareis
import numpy as np
import open3d as o3d
import trimesh

# loading DEM
dem = np.load("dem_test.npy")

lon = dem[:, 0]
lat = dem[:, 1]
elev = dem[:, 2]

#equirectangular (plate carrée) approximation again
lat0 = lat.mean()
meters_lat = 111320
meters_lon = 111320 * np.cos(np.radians(lat0))

# Getting x, y, z
x = (lon - lon.mean()) * meters_lon
y = (lat - lat.mean()) * meters_lat
z = elev


# function for making a mesh using trimesh
def make_mesh(x, y, z, rows, cols):
    # getting vericies
    verts = np.column_stack([x, y, z])

    a, b = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    # each quad, compute the flat vertex indices of the four corners
    v00 = (a * cols + b).ravel()
    v01 = (a * cols + (b + 1)).ravel()
    v10 = ((a + 1) * cols + b).ravel()
    v11 = ((a + 1) * cols + (b + 1)).ravel()
    
    #split each quad into two triangles
    faces_1 = np.column_stack([v00, v10, v01])
    faces_2 = np.column_stack([v01, v10, v11])
    faces = np.vstack([faces_1, faces_2])

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# making mesh with function
mesh = make_mesh(x, y, z, rows=635, cols=1627)
#mesh.export("dem_mesh.ply")

# using open3d to visualise the mesh
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
o3d_mesh.compute_vertex_normals()

o3d.visualization.draw_geometries([o3d_mesh])

