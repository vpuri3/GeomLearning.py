#
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

__all__ = [
    'visualize_mpl',
    'visualize_o3d',
    'make_trimesh',
]

def visualize_mpl(graph, make_edge=False, cmap='jet'):
    pos = graph.x.numpy(force=True)
    val = graph.y.numpy(force=True)
    # elm = graph.elems.numpy(force=True)
    edx = graph.edge_index.numpy(force=True).T

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
        for edge in edx:
            start, end = edge
            ax.plot([pos[start][0], pos[end][0]],
                    [pos[start][1], pos[end][1]],
                    [pos[start][2], pos[end][2]],
                    c='gray', linewidth=0.5)

    return fig

def visualize_o3d(graph, imagefile):
    pos = graph.x.numpy(force=True)
    val = graph.y.numpy(force=True)
    elm = graph.elems.numpy(force=True)
    # edx = graph.edge_index.numpy(force=True)

    mesh = make_trimesh(pos, elm, val)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    # vis.poll_events()
    # vis.update_renderer()
    vis.capture_screen_image(imagefile)
    vis.destroy_window()

    return

def make_trimesh(verts, elems, values, cmap='jet'):
    tri_idx = np.array([[0,0,0,0,0,0,6,6,6,6,6,6],
                        [2,3,7,4,1,5,2,3,1,5,4,7],
                        [1,2,3,7,5,4,3,7,2,1,5,4]], dtype=int)
    mesh = o3d.geometry.TriangleMesh()
    tris = np.zeros([12*elems.shape[0],3]).astype(np.int32)
    for i, elem in enumerate(elems):
        tris[i*12:(i+1)*12,:] = elem[tri_idx].T#-1

    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))

    cmap = plt.get_cmap(cmap)
    lb, ub = min(values), max(values)
    values = (values - lb) / (ub - lb)
    colors = cmap(values.reshape(-1)) # [Nv, 4]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:,:3].astype(float))

    return mesh

