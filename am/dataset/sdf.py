#
import torch
import numpy as np
from scipy.spatial import KDTree

__all__ = [
    'compute_distance_to_surface',
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

def extract_surface_nodes(pos, elems):
    """
    Extract surface nodes from a 3D hexahedral mesh
    """

    # Compute all faces
    faces = make_faces(elems) # [Nfaces, 4]
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    # Get indices of faces that appear exactly once
    faces = faces[unique_indices[counts == 1]]
    
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

    surface_faces = faces[is_surface]
    surface_nodes = np.unique(surface_faces.reshape(-1))
    surface_pos = pos[surface_nodes]

    return surface_pos

#======================================================================#
def compute_distance_to_surface(pos, elems):
    """
    Compute direction vectors from each node to nearest surface point
    Args:
        pos: (N, 3) array of node positions
        elems: (E, 8) array of hex element connectivity
    Returns:
        directions: (N, 3) array of direction vectors (x, y, z components)
    """

    pos = pos.numpy(force=True)
    elems = elems.numpy(force=True)

    # extract surface nodes
    surface_pos = extract_surface_nodes(pos, elems)
    # Build KDTree for surface pos
    tree = KDTree(surface_pos) # reduce leafsize
    # Find nearest surface points and their indices
    distances, indices = tree.query(pos)
    # Get the nearest surface points
    nearest_points = surface_pos[indices]
    # Compute direction vectors from nodes to nearest surface points
    directions = nearest_points - pos
    
    err = np.linalg.norm(directions, 2, axis=-1) - distances
    assert np.max(err) < 1e-5, f"max error: {np.max(err)}, avg error: {np.mean(err)}"
    
    return torch.tensor(directions, dtype=torch.float)

#======================================================================#
#