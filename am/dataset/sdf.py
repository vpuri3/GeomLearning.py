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
    
    # face edge sizes
    L = np.linalg.norm(pos[faces[:, 0]] - pos[faces[:, 1]], axis=1)
    W = np.linalg.norm(pos[faces[:, 1]] - pos[faces[:, 2]], axis=1)
    
    # exclude faces at T-Junctions
    tree = KDTree(face_centers)

    # Create a mask for surface faces (initially all True)
    is_surface = np.ones(len(faces), dtype=bool)
    
    # For each face, find nearby faces and check if they're opposite
    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
        if not is_surface[i]:
            continue
            
        # if truly a T-junction:
        # Find faces within a distance based on face size
        # we are grabbing neighbors only for the large face at a T-junction
        face_size = np.sqrt(L[i]**2 + W[i]**2) / 4
        neighbors = tree.query_ball_point(center, r=face_size * 1.001)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or not is_surface[j]:
                continue
                
            # Check if normals align
            # ideally normals should be opposite, but orientation isn't guaranteed
            if not np.abs(np.dot(normal, face_normals[j])) > 0.999:
                continue
            
            # get direction bw face centers
            # if direction is axis-aligned, then continue
            face_center_disp = face_centers[j] - center
            face_center_dist = np.linalg.norm(face_center_disp)
            face_center_dirn = face_center_disp / face_center_dist
            if np.abs(face_center_dirn[0]) > 0.999 or np.abs(face_center_dirn[1]) > 0.999 or np.abs(face_center_dirn[2]) > 0.999:
                continue
            
            if face_center_dist < face_size * 0.999:
                continue
            
            # these faces are T-junctions, not surface
            is_surface[i] = False
            is_surface[j] = False

    return faces[is_surface]

def extract_surface_faces(pos, elems):
    """
    Extract surface nodes from a 3D hexahedral mesh
    """

    # Get all faces
    faces = make_faces(elems) # [Nfaces, 4]
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    # Get indices of faces that appear exactly once
    unique_faces = faces[unique_indices[counts == 1]]
    # omit faces at T-Junctions
    surface_faces = omit_tjunction_faces(unique_faces, pos)

    return surface_faces

def create_surface_trimesh(pos, elems):
    """
    Create a surface trimesh from a 3D hexahedral mesh
    """
    # extract surface nodes
    surface_faces = extract_surface_faces(pos, elems)

    # get surface nodes
    surface_indices = np.unique(surface_faces.reshape(-1))
    surface_pos = pos[surface_indices]
    
    # create triangles from surface faces
    surface_triangles = quads_to_triangles(surface_faces)

    # remap surface_triangles to only index surface nodes
    vertex_remap = np.full(len(pos), -1, dtype=int)
    vertex_remap[surface_indices] = np.arange(len(surface_indices))
    surface_triangles = vertex_remap[surface_triangles]

    # create trimesh object
    surface_mesh = trimesh.Trimesh(vertices=surface_pos, faces=surface_triangles)

    return surface_mesh, surface_indices

def surface_ray_intersect(surface_mesh, ray_origins, ray_direction, interior=False):
    """
    Ray cast to surface
    """
    assert ray_origins.ndim == 2 and ray_origins.shape[1] == 3
    assert ray_direction.ndim == 1 and len(ray_direction) == 3

    ray_directions = np.repeat(ray_direction.reshape(1,3), len(ray_origins), axis=0)

    # get nearest point on surface
    intersect_locs, ray_indices, tri_indices = surface_mesh.ray.intersects_location(
        ray_origins,
        ray_directions,
        multiple_hits=False,
    )
    
    if interior:
        assert len(intersect_locs) == len(ray_origins), f'Ray cast error: {len(intersect_locs)} != {len(ray_origins)}'
        return intersect_locs
    
    # get ray intersection points
    intersect_locations = np.full((len(ray_origins), 3), -9999.)
    # intersect_locations = ray_origins.copy()
    
    # print(f"num query: {len(ray_origins)}")
    # print(f"num hits : {len(intersect_locs)}")
    
    if len(intersect_locs) > 0:
        intersect_locations[ray_indices] = intersect_locs

    return intersect_locations

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

    surface_mesh, surface_indices = create_surface_trimesh(pos, elems)

    # get nearest point on surface
    nearest_point, _, _ = surface_mesh.nearest.on_surface(pos)
    nearest_directions = nearest_point - pos

    # # get overhang distances
    # # intuition: how much material is under me before thin air?
    # # not correct at surface nodes
    # overhang_intersections = surface_ray_intersect(surface_mesh, pos, np.array([0., 0., -1.]))
    # overhang_distances = (pos[:,2] - overhang_intersections[:,2]).reshape(-1,1)

    # for surface:
    # 1. Alternative: compute true distance to surface for interior nodes
    # 2. get value from nearest surface node and add difference in position
    
    # get surface mask
    surface_mask = np.zeros(len(pos), dtype=bool)
    surface_mask[surface_indices] = True
    
    # get overhang distances for interior nodes
    intersections_interior = surface_ray_intersect(
        surface_mesh,
        pos[~surface_mask],
        np.array([0., 0., -1.]),
        # interior=True,
        interior=False,
    )
    overhang_distances_interior = pos[~surface_mask, 2] - intersections_interior[:, 2]
    
    # surface
    # overhang_distances_surface = np.zeros(np.sum(surface_mask))
    overhang_distances_surface = np.full((np.sum(surface_mask),), 9999.)
    
    # combine
    overhang_distances = np.full((len(pos),), 0)
    overhang_distances[surface_mask] = overhang_distances_surface
    overhang_distances[~surface_mask] = overhang_distances_interior
    overhang_distances = overhang_distances.reshape(-1,1)

    #----------------------------------#
    return np.hstack([
        nearest_directions,
        overhang_distances,
        surface_mask.reshape(-1,1),
    ])

#======================================================================#
#