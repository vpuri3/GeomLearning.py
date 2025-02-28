#
# installation on MAIL lab clusters
# - CUDA Driver 11.1
# - glibc version: ($ ldd --version) ldd (GNU libc) 2.17)
#

conda create --name GeomLearning python=3.9
conda create --name GeomLearningCPU python=3.11

#-------------------#
# GeomLearning
#-------------------#
conda activate GeomLearning
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install "numpy<2"

pip install torch-geometric==2.2.0 # (2.2.3)
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl

pip install ipython
pip install tqdm jsonargparse timm einops
pip install scipy pandas seaborn pyvista matplotlib # trimesh rtree

#-------------------#
# GeomLearningCPU
#-------------------#
conda activate GeomLearningCPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tqdm torch_geometric jsonargparse timm einops
pip install scipy pandas seaborn pyvista matplotlib trimesh rtree
conda install -c conda-forge -c open3d-admin open3d
#
