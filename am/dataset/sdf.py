#
import trimesh # also needs package rtree
import numpy as np
from scipy.spatial import KDTree


__all__ = [
    'distance_to_surface',
]

#======================================================================#
def make_faces(elems):
    """
    Get all faces (6 per hex, 4 nodes per face)
    """
    assert elems.ndim == 2 and elems.shape[1] == 8

    faces = np.hstack([
        elems[:, [0, 1, 2, 3]],  # bottom
        elems[:, [4, 5, 6, 7]],  # top
        elems[:, [0, 1, 5, 4]],  # front
        elems[:, [1, 2, 6, 5]],  # right
        elems[:, [2, 3, 7, 6]],  # back
        elems[:, [3, 0, 4, 7]]   # left
    ])

    faces = faces.reshape(-1, 4)
    return faces

def quads_to_triangles(faces):
    """
    Convert quads to triangles
    """
    assert faces.ndim == 2 and faces.shape[1] == 4

    triangles = np.vstack([
        faces[:, [0, 1, 2]],
        faces[:, [0, 2, 3]]
    ])

    return triangles

def omit_tjunction_faces(faces, pos):
    """
    Omit faces at T-Junctions
    """

    # face centers and normals
    face_centers = np.mean(pos[faces], axis=1)
    v1 = pos[faces[:, 1]] - pos[faces[:, 0]]
    v2 = pos[faces[:, 2]] - pos[faces[:, 0]]
    face_normals = np.cross(v1, v2)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    
    # exclude faces at T-Junctions
    tree = KDTree(face_centers)

    # Create a mask for surface faces (initially all True)
    is_surface = np.ones(len(faces), dtype=bool)
    
    # For each face, find nearby faces and check if they're opposite
    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
        if not is_surface[i]:
            continue
            
        # Find faces within a distance based on face size
        # since r_factor < 1, we are grabbing neighbors only for the large face at a T-junction
        face_size = np.max(np.linalg.norm(pos[faces[i]] - center, axis=1))
        neighbors = tree.query_ball_point(center, r=face_size * 0.75)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or not is_surface[j]:
                continue
                
            # Check if normals align
            # ideally they should be opposite, but orientation is not guaranteed
            dot = np.dot(normal, face_normals[j])
            
            if np.abs(dot) > 0.99:
                is_surface[i] = False
                is_surface[j] = False

    return faces[is_surface]

def extract_surface_faces(pos, elems):
    """
    Extract surface nodes from a 3D hexahedral mesh
    """

    # Compute all faces
    faces = make_faces(elems) # [Nfaces, 4]
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    # Get indices of faces that appear exactly once
    unique_faces = faces[unique_indices[counts == 1]]
    # omit faces at T-Junctions
    surface_faces = omit_tjunction_faces(unique_faces, pos)

    return surface_faces

#======================================================================#
def distance_to_surface(pos, elems):
    """
    Compute distance to surface
    Args:
        pos: (N, 3) array of node positions
        elems: (E, 8) array of hex element connectivity
    Returns:
        directions: (N, 3) array of direction vectors (x, y, z components)
    """

    pos = pos.numpy(force=True)
    elems = elems.numpy(force=True)

    # extract surface nodes
    surface_faces = extract_surface_faces(pos, elems)

    #-------------------------------#
    # KDTree method
    #-------------------------------#
    # get surface nodes
    surface_indices = np.unique(surface_faces.reshape(-1))
    surface_pos = pos[surface_indices]

    # # Build KDTree for surface pos
    # tree = KDTree(surface_pos)
    # # Find nearest surface points and their indices
    # _, indices = tree.query(pos)
    # # Get the nearest surface points
    # nearest_points = surface_pos[indices]
    # # Compute direction vectors from nodes to nearest surface points
    # nearest_directions = nearest_points - pos
    # return nearest_directions
    #-------------------------------#
    
    # create triangles from surface faces
    surface_triangles = quads_to_triangles(surface_faces)

    # remap surface_triangles to only index surface nodes
    vertex_remap = np.full(len(pos), -1, dtype=int)
    vertex_remap[surface_indices] = np.arange(len(surface_indices))
    surface_triangles = vertex_remap[surface_triangles]

    # create trimesh object
    surface_mesh = trimesh.Trimesh(vertices=surface_pos, faces=surface_triangles)
    
    # get nearest point on surface
    nearest_point, _, _ = surface_mesh.nearest.on_surface(pos)
    nearest_directions = nearest_point - pos

    # # compute overhang size
    # overhang_directions = np.zeros((len(pos), 3))
    # overhang_directions[:, 2] = -1
    # overhang_locs, _, _ = surface_mesh.ray.intersects_location(pos, overhang_directions)
    
    # # Initialize overhang_distances with large values (no intersection)
    # overhang_distances = np.full(len(pos), 999.)
    
    # # For intersecting rays, compute distances
    # if len(overhang_locs) > 0:
    #     # Find which rays actually intersected
    #     _, idx = surface_mesh.ray.intersects_id(pos, overhang_directions, return_locations=False)
    #     # Compute distances for intersecting rays
    #     overhang_distances[idx] = np.linalg.norm(overhang_locs - pos[idx], axis=-1)

    return np.hstack([
        nearest_directions,
        # overhang_distances,
    ])

#======================================================================#

#======================================================================#
#