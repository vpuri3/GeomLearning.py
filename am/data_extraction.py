#
import numpy as np
from tqdm import tqdm

import os
import struct
import zipfile
import shutil


__all__ = [
    "extract",
]

#=================================#
# Based on Andrew Porco's
# https://github.com/andrewporco/netfabb//netfabb-tools/extract-stress-strain.py
# which is in turn based on Kevin Ferguson's
# https://github.com/kevinferg/netfabb-tools/extract-netfabb-data-binary.py
#=================================#

#=================================#
# read files
#=================================#

def get_basefile(full_path):
    return os.path.normpath(os.path.basename(full_path))

def extract_frames_from_case(filename):
    with open(filename,'r') as f:
        lines = f.readlines()

    for line in lines:
        if "number of steps:" in line:
            return int(line[17:])

    return []

def read80(f):
    return f.read(80).decode('utf-8').strip()

def read_floats(f,N):
    arr = np.array(struct.unpack(f"<{N}f", f.read(4*N)))
    return arr[0] if N == 1 else arr

def read_ints(f,N):
    arr = np.array(struct.unpack(f"<{N}i", f.read(4*N)))
    return arr[0] if N == 1 else arr

def read_geo_binary(path):
    with open(path, 'rb') as f:

        ############ Info
        assert read80(f) == 'Fortran Binary'
        description1 = read80(f)
        description2 = read80(f)
        assert read80(f) == 'node id off'
        assert read80(f) == 'element id off'
        # extents_str = read80(f)       # No extents
        # extents = read_floats(f,6)    # No extents

        ############ Begin part 1
        assert read80(f) == 'part'
        assert read_ints(f,1) == 1 # Should be 1 part only
        description3 = read80(f)

        ############ Coordinates
        assert read80(f) == 'coordinates'
        nn = read_ints(f,1)
        # node_ids = read_ints(f,nn) # node id is off
        x = read_floats(f,nn)
        y = read_floats(f,nn)
        z = read_floats(f,nn)
        nodes = np.vstack([x,y,z]).T

        ############ Elements
        element_type = read80(f)
        assert(element_type == 'hexa8')
        # element_ids = read_ints(f,nn) # element id is off
        ne = read_ints(f, 1)
        elems = read_ints(f, 8*ne).reshape(ne,8)

    data = dict(description1=description1, description2=description2, description3=description3, 
                nn=nn, nodes=nodes, element_type=element_type, ne=ne, elems=elems)
    return data


def read_ens_binary(path, num_nodes, num_values):
    with open(path, 'rb') as f:
        description = read80(f)
        assert(read80(f) == 'part')
        assert(read_ints(f, 1) == 1)
        assert(read80(f) == 'coordinates')
        arr = read_floats(f, num_nodes * num_values)
    data = arr.reshape(num_values, num_nodes).T
    return dict(description=description, data=data)

def get_vertices_from_geo(filename, return_elements=False):
    data = read_geo_binary(filename)
    # print(data["nodes"], data["elems"])
    if not return_elements:
        return data["nodes"]
    else:
        return data["nodes"], data["elems"]

def get_values_from_ens(filename, N, nv):
    data = read_ens_binary(filename, N, nv)
    return data["data"]

#=================================#
# grab results
# modify only: get_file_info, get_all_results, extract_data
#=================================#
def get_file_info(filename):
    basename = get_basefile(filename) + "_"
    if not os.path.exists(f"{filename}/results/{basename}mechanical.case"):
        basename = ""
        if not os.path.exists(f"{filename}/results/mechanical.case"):
            return dict(error="Error preparing simulation.")
    with open(f'{filename}/{basename}mechanical.out') as f:
        if 'Analysis terminated' in f.read():
            return dict(error="Error running simulation.")
    case_file = f"{filename}/results/{basename}mechanical.case"

    frame_count = extract_frames_from_case(case_file)
    if frame_count == []:
        return dict(error=f"Case file missing frame count. {filename}")

    semi_frame_count = frame_count//2+1
    geo_file = f"{filename}/results/{basename}mechanical_{semi_frame_count}.geo"
    if not os.path.exists(geo_file):
        return dict(error="Missing Geo file.")
    basefilename = f"{filename}/results/{basename}mechanical00_{frame_count}."

    return dict(geo=geo_file, basefilename=basefilename)

def get_all_results(filename):
    info = get_file_info(filename)
    if len(info) == 1:
        return [info["error"],] # Simulation failed
    
    verts, elems = get_vertices_from_geo(info["geo"], return_elements=True)
    N_verts = verts.shape[0]

    result_types = dict(dis="disp", ept="strain", rcd="recoater_status", rct="recoater_clearance_percent", sd1="max_dir", sd2="mid_dir", sd3="min_dir",
                        sig="cauchy_stress", sp1="max_stress", sp2="mid_stress", sp3="min_stress", svm="von_mises_stress", tmp="temp")
    result_nums = dict(dis=3, ept=6, rcd=1, rct=1, sd1=3, sd2=3, sd3=3, sig=6, sp1=1, sp2=1, sp3=1, svm=1, tmp=1)

    results = dict(verts=verts, elems=elems)
    for key in result_types:
        result = get_values_from_ens(info['basefilename'] + key + ".ens",N_verts, result_nums[key])
        results[result_types[key]] = result.astype(np.float32)

    return results

def extract_data(data_dir, out_dir, error_file):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    cases = os.listdir(data_dir)
    print(f"Loading displacement results from: {data_dir} into {out_dir}")

    num_success = 0
    num_failure = 0

    with open(error_file,"a") as err:
        for case in tqdm(cases):
            casefile = os.path.join(data_dir, case)
            if os.path.isdir(casefile):
                results = get_all_results(casefile)
                if len(results) == 1:
                    num_failure += 1
                    err.write(f'{case}\n')
                else:
                    num_success += 1
                    output_path = os.path.join(out_dir, case + '.npz')
                    np.savez(output_path, **results)

    print(f"Successfully saved {num_success} / {num_success + num_failure} cases to NPZ format.")
    print(f"Failed simulation cases are logged to {error_file}")

#=================================#
# assemble data
#=================================#
def unzip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(extract_dir)

def extract(source_dir, target_dir):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    zipfiles = [f for f in os.listdir(source_dir) if f.endswith('.zip')]
    for filename in zipfiles:
        # extract zip file
        zip_path = os.path.join(source_dir, filename)
        extract_dir = os.path.join(target_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"Unzipping {filename}...")
        # unzip(zip_path, extract_dir)

        # get data
        data_dir = os.path.join(extract_dir, "SandBox")
        out_dir  = os.path.join(target_dir, filename[:-4])
        err_file = os.path.join(out_dir, "error.txt")
        extract_data(data_dir, out_dir, err_file)

        # clean up
        print(f"Cleaning up extracted file: {extract_dir}")
        # shutil.rmtree(extract_dir)
        break
    return

#=================================#
#