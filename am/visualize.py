#
import torch
import numpy as np
import torch_geometric as pyg

__all__ = [
    'mesh_pyv',
    'write_pvd',
    'visualize_mpl',
    'visualize_o3d',
    'visualize_tri',
]

#======================================================================#
def tri_faces(elems):
    # break up 6 faces of an hexa8 element into 12 triangles
    tri_idx = np.array([[0,0,0,0,0,0,6,6,6,6,6,6],
                        [2,3,7,4,1,5,2,3,1,5,4,7],
                        [1,2,3,7,5,4,3,7,2,1,5,4]], dtype=int)

    tri_faces = np.zeros([12*elems.shape[0],3], dtype=int)
    for i, elem in enumerate(elems):
        tri_faces[i*12:(i+1)*12,:] = elem[tri_idx].T

    return tri_faces

def quad_faces(elems):
    # break out the 6 faces of an hexa8 element
    quad_idx = np.array([[0,4,0,1,2,3,],
                         [1,5,1,2,3,0,],
                         [2,6,5,6,7,4,],
                         [3,7,4,5,6,7,]], dtype=int)

    quad_faces = np.zeros([6*elems.shape[0],4], dtype=int)
    for i, elem in enumerate(elems):
        quad_faces[i*12:(i+1)*12,:] = elem[quad_idx].T

    return quad_faces

#======================================================================#
def mesh_pyv(pos: torch.Tensor, elems: torch.Tensor):
    import pyvista as pv

    PPE = 8 # points per element
    celltype = pv.CellType.HEXAHEDRON
    celltypes = np.full(elems.shape[0], celltype, dtype=np.uint8)

    pos = pos.numpy(force=True)   # [Nv, 3]
    elm = elems.numpy(force=True) # [Ne, PPE]

    NV = pos.shape[0]
    NE = elm.shape[0]

    points = pos
    cells  = np.concat((np.full((NE,1), PPE), elm), axis=1)#.ravel()
    mesh   = pv.UnstructuredGrid(cells, celltypes, points)

    return mesh

def write_pvd(pvd_file, N, vtu_name):
    with open(pvd_file, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for i in range(N):
            f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vtu_name}{str(i).zfill(2)}.vtu"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    return

#======================================================================#
def visualize_mpl(
    pos: torch.Tensor, val: torch.Tensor, edge_index: torch.Tensor,
    make_edge=True, max_edges=100_000, cmap='jet'
):
    import matplotlib.pyplot as plt

    pos = pos.numpy(force=True)
    val = val.numpy(force=True)
    edx = edge_index.sort(dim=0).values.unique(dim=1).numpy(force=True).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cmap = plt.get_cmap(cmap)
    lb, ub = min(val), max(val)
    cvals = (val - lb) / (ub - lb)
    colors = cmap(cvals.reshape(-1)) # [Nv, 4]

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, s=2)
    if make_edge:
        for (i, edge) in enumerate(edx):
            if i >= max_edges:
                print(f"Plotting {max_edges} / {edx.shape[0]} edges.")
                break
            start, end = edge
            ax.plot([pos[start][0], pos[end][0]],
                    [pos[start][1], pos[end][1]],
                    [pos[start][2], pos[end][2]],
                    c='gray', linewidth=0.5)

    return fig

#======================================================================#
def visualize_o3d(
    pos: torch.Tensor, val: torch.Tensor, elems: torch.Tensor,
    imagefile: str,
):
    import open3d as o3d

    pos = pos.numpy(force=True)
    val = val.numpy(force=True)
    elm = elems.numpy(force=True)

    # MESH
    tris = tri_faces(elems)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))

    cmap = plt.get_cmap(cmap)
    lb, ub = min(values), max(values)
    values = (values - lb) / (ub - lb)
    colors = cmap(values.reshape(-1)) # [Nv, 4]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:,:3].astype(float))

    # VIS
    o3d.visualization.webrtc_server.enable_webrtc()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.add_geometry(mesh)
    # # vis.poll_events()
    # # vis.update_renderer()
    # vis.capture_screen_image(imagefile)
    # vis.destroy_window()

    return

#======================================================================#
def visualize_tri(
    pos: torch.Tensor, val: torch.Tensor, elems: torch.Tensor,
    imagefile: str, cmap='jet'
):
    import trimesh
    import matplotlib.pyplot as plt

    elems = elems.numpy(force=True)
    faces = tri_faces(elems)

    graph = pyg.data.Data(pos=pos, y=val, face=torch.tensor(faces))
    mesh  = pyg.utils.to_trimesh(graph)

    cmap = plt.get_cmap(cmap)
    lb, ub = min(val), max(val)
    cvals = (val - lb) / (ub - lb)
    colors = cmap(cvals.reshape(-1)) # [Nv, 4]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
        triangles=mesh.faces,
        cmap='jet',
        edgecolor='black', linewidth=0.2,
    )

    return fig
#======================================================================#
#
