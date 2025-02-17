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

#======================================================================#
def select_tjt_large(faces, pos):
    """
    Select large faces at T-Junctions
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

    # Create a mask for large faces at T-Junctions (initially all False)
    is_tjt_large = np.zeros(len(faces), dtype=bool)
    
    # For each face, find nearby faces and check if they're opposite
    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):

        # Find faces within a distance based on expected center distance in a T-junction
        # we are grabbing neighbors only for the large face at a T-junction
        tjt_center_dist = np.sqrt(L[i]**2 + W[i]**2) / 4
        neighbors = tree.query_ball_point(center, r=tjt_center_dist * 1.001)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or is_tjt_large[j]:
                continue
                
            # Check if normals align
            # ideally normals should be opposite, but orientation isn't guaranteed
            if not np.abs(np.dot(normal, face_normals[j])) > 0.999:
                continue
            
            face_center_disp = face_centers[j] - center
            face_center_dist = np.linalg.norm(face_center_disp)

            # if differs from exected tjt_center_dist then it is not a T-junction
            if face_center_dist < tjt_center_dist * 0.999:
                continue
            
            # get direction bw face centers
            # if direction is axis-aligned, then continue
            face_center_direction = face_center_disp / face_center_dist
            if (
                np.abs(face_center_direction[0]) > 0.999 or
                np.abs(face_center_direction[1]) > 0.999 or
                np.abs(face_center_direction[2]) > 0.999
            ):
                continue
            
            # if we are here, then face [i] is a large face at a T-junction
            is_tjt_large[i] = True

    return is_tjt_large

def create_point_hash(pos):
    """
    Create a hash table for point coordinates using integer coordinates
    Args:
        pos: (N, 3) array of point coordinates
    Returns:
        point_hash: dictionary mapping tuple of rounded coordinates to indices
    """
    # Scale and round coordinates to integers
    scale = 1e6  # adjust based on required precision
    int_pos = np.round(pos * scale).astype(np.int64)
    
    # Create hash using tuples of integer coordinates
    point_hash = {tuple(coords): idx for idx, coords in enumerate(int_pos)}
    return point_hash

def check_point_existence(point_hash, point):
    """
    Check if a point exists in the hash table
    Args:
        point_hash: dictionary from create_point_hash
        point: (3,) array of point coordinates
    Returns:
        index of point if exists, None otherwise
    """
    # Scale and round the point to match hash format
    scale = 1e6  # must match scale used in create_point_hash
    int_point = tuple(np.round(point * scale).astype(np.int64))
    return point_hash.get(int_point, None)

def break_tjunctions(faces, pos, debug=None):
    """
    Break faces at T-Junctions
    """
    
    hash = create_point_hash(pos)

    # get large faces at T-Junctions
    is_tjt_large = select_tjt_large(faces, pos)

    # break faces at T-Junctions
    faces_new = faces[~is_tjt_large]
    pos_new = pos.copy()

    # break face at T-Junction
    for face in faces[is_tjt_large]:

        # get positions
        p0, p1, p2, p3 = pos[face]
        p4 = np.mean(pos[face[[0, 1]]], axis=0) # get edge centers
        p5 = np.mean(pos[face[[1, 2]]], axis=0)
        p6 = np.mean(pos[face[[2, 3]]], axis=0)
        p7 = np.mean(pos[face[[3, 0]]], axis=0)
        p8 = np.mean(pos[face], axis=0) # get face center

        # get indices
        i0, i1, i2, i3 = face
        # check if p4-p7 already exist in pos.
        i4 = check_point_existence(hash, p4)
        i5 = check_point_existence(hash, p5)
        i6 = check_point_existence(hash, p6)
        i7 = check_point_existence(hash, p7)
        i8 = check_point_existence(hash, p8)

        if i4 is None:
            pos_new = np.vstack([pos_new, p4])
            i4 = len(pos_new) - 1
        if i5 is None:
            pos_new = np.vstack([pos_new, p5])
            i5 = len(pos_new) - 1
        if i6 is None:
            pos_new = np.vstack([pos_new, p6])
            i6 = len(pos_new) - 1
        if i7 is None:
            pos_new = np.vstack([pos_new, p7])
            i7 = len(pos_new) - 1
        if i8 is None:
            pos_new = np.vstack([pos_new, p8])
            i8 = len(pos_new) - 1
    
        # get new faces
        faces_new = np.vstack([
            faces_new,
            np.array([i0, i4, i8, i7]),
            np.array([i4, i1, i5, i8]),
            np.array([i8, i5, i2, i6]),
            np.array([i7, i8, i6, i3]),
        ])

    if debug is not None:
        print(f"break_tjunctions: added {len(faces_new) - len(faces)}/{len(faces)} faces, {len(pos_new) - len(pos)}/{len(pos)} nodes")
    
    return pos_new, faces_new

def omit_internal_tjunctions(faces, pos, debug=None):
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
            
        # Find faces within a distance based on expected center distance in a T-junction

        tjt_center_dist = np.sqrt(L[i]**2 + W[i]**2) / 4
        # we are grabbing neighbors only for the large face at a T-junction
        neighbors = tree.query_ball_point(center, r=tjt_center_dist * 1.001)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or not is_surface[j]:
                continue
                
            # Check if normals align
            # ideally normals should be opposite, but orientation isn't guaranteed
            if not np.abs(np.dot(normal, face_normals[j])) > 0.999:
                continue
            
            face_center_disp = face_centers[j] - center
            face_center_dist = np.linalg.norm(face_center_disp)

            # if differs from exected tjt_center_dist then it is not a T-junction
            if face_center_dist < tjt_center_dist * 0.999:
                continue
            
            # get direction bw face centers
            # if direction is axis-aligned, then continue
            face_center_direction = face_center_disp / face_center_dist
            if (
                np.abs(face_center_direction[0]) > 0.999 or
                np.abs(face_center_direction[1]) > 0.999 or
                np.abs(face_center_direction[2]) > 0.999
            ):
                continue
            
            # these faces are T-junctions, not surface
            is_surface[i] = False
            is_surface[j] = False

    if debug is not None:
        print(f"omit_internal_tjunctions: omitted {np.sum(~is_surface)}/{len(faces)} faces")
    
    return faces[is_surface]

def create_surface_trimesh(pos, elems, debug=None):
    """
    Create a surface trimesh from a 3D hexahedral mesh
    """

    ###
    # extract surface nodes
    ###

    # Get all faces
    faces = make_faces(elems) # [Nfaces, 4]
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    unique_faces = faces[unique_indices[counts == 1]]
    # omit faces at internal T-Junctions
    surface_faces = omit_internal_tjunctions(unique_faces, pos)
    # Break remaining T-junctions
    pos_new, surface_faces = break_tjunctions(surface_faces, pos, debug=debug)
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(surface_faces, axis=1), axis=0, return_counts=True, return_index=True)
    surface_faces = surface_faces[unique_indices[counts == 1]]

    ###
    # get surface triangles and nodes
    ###

    # surface nodes
    surface_indices = np.unique(surface_faces.reshape(-1))
    surface_pos = pos_new[surface_indices]
    
    # surface triangles
    surface_triangles = quads_to_triangles(surface_faces)
    # remap surface_triangles to only index surface nodes
    vertex_remap = np.full(len(pos_new), -1, dtype=int)
    vertex_remap[surface_indices] = np.arange(len(surface_indices))
    surface_triangles = vertex_remap[surface_triangles]

    ###
    # create trimesh object
    ###
    surface_mesh = trimesh.Trimesh(vertices=surface_pos, faces=surface_triangles)
    surface_indices = surface_indices[surface_indices < len(pos)]

    return surface_mesh, surface_indices[surface_indices < len(pos)]

def surface_ray_intersect(surface_mesh, ray_origins, ray_direction, debug=None):
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
    
    if len(intersect_locs) < len(ray_origins):
        if debug is not None:
            print(f"In surface_ray_intersect:")
            print(f"num query: {len(ray_origins)}")
            print(f"num hits : {len(intersect_locs)}")
    
        intersect_locations = np.full((len(ray_origins), 3), np.nan)
        if len(intersect_locs) > 0:
            intersect_locations[ray_indices] = intersect_locs
        return intersect_locations
    else:
        return intersect_locs
    
def dist_to_surface_in_dir(direction, surface_mesh, surface_mask, pos, tree_interior=None, debug=None):
    """
    Compute distance to surface in a given direction
    """
    direction = direction / np.linalg.norm(direction)
    
    intersections_interior = surface_ray_intersect(surface_mesh, pos[~surface_mask], direction, debug=debug)
    
    # deal with missed intersections
    isnan_interior = np.isnan(intersections_interior.sum(axis=1))
    
    if np.any(isnan_interior):
        tree_interior_nan = KDTree(pos[~surface_mask][~isnan_interior])
        _, nearest_indices_nan_interior = tree_interior_nan.query(pos[~surface_mask][isnan_interior], k=1)
        difference_nan_interior = pos[~surface_mask][isnan_interior] - pos[~surface_mask][~isnan_interior][nearest_indices_nan_interior]
        intersections_interior[isnan_interior] = intersections_interior[~isnan_interior][nearest_indices_nan_interior] + difference_nan_interior

    # deal with surface
    if tree_interior is None:
        tree_interior = KDTree(pos[~surface_mask])
    
    _, nearest_indices_surface = tree_interior.query(pos[surface_mask], k=1)
    difference_surface = pos[surface_mask] - pos[~surface_mask][nearest_indices_surface]
    intersections_surface = intersections_interior[nearest_indices_surface] + difference_surface
    
    # assemble intersections array
    intersections = np.zeros((len(pos), 3,))
    intersections[~surface_mask] = intersections_interior
    intersections[surface_mask] = intersections_surface

    # compute distances
    distances = np.abs(np.dot(pos - intersections, direction))
    
    assert not np.any(np.isnan(distances)), "dist_to_surface_in_dir: Found NaN values in distances array"

    return distances

#======================================================================#
def distance_to_surface(pos, elems, debug=None):
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

    surface_mesh, surface_indices = create_surface_trimesh(pos, elems, debug=debug)

    ###
    # get SDF direction and magnitude
    ###
    nearest_point, _, _ = surface_mesh.nearest.on_surface(pos)
    sdf_direction = nearest_point - pos
    sdf_magnitude = np.linalg.norm(sdf_direction, axis=1)

    ###
    # get surface mask
    ###
    surface_mask = np.zeros(len(pos), dtype=bool)
    surface_mask[surface_indices] = True
    
    ###
    # misc
    ###
    tree_interior = None
    tree_interior = KDTree(pos[~surface_mask])

    # overhang_distances: how much material is under me before thin air?
    overhang_distances = dist_to_surface_in_dir(
        np.array([0., 0., -1.]),
        surface_mesh,
        surface_mask,
        pos,
        tree_interior=tree_interior,
        debug=debug,
    )
    
    return np.hstack([
        surface_mask.reshape(-1,1),
        sdf_magnitude.reshape(-1,1),
        overhang_distances.reshape(-1,1),
        sdf_direction,
    ])

#======================================================================#
#