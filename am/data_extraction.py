#
import torch
import numpy as np
from tqdm import tqdm

import os
import struct
import zipfile
import shutil
import collections

__all__ = [
    'extract_zips',
    'extract_from_dir',
    'extract_from_zip',
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

def extract_frames_from_case(casefile): # mechanical.case
    with open(casefile,'r') as f:
        lines = f.readlines()

    nframes = []
    for line in lines:
        if "number of steps:" in line:
            fr = int(line[17:])
            nframes.append(fr)

    return nframes

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


def read_ens_binary(path, num_nodes, dim):
    with open(path, 'rb') as f:
        description = read80(f)
        assert(read80(f) == 'part')
        assert(read_ints(f, 1) == 1)
        assert(read80(f) == 'coordinates')
        arr = read_floats(f, num_nodes * dim)
    data = arr.reshape(dim, num_nodes).T
    return dict(description=description, data=data)

def get_vertices_from_geo(filename, return_elements=False):
    data = read_geo_binary(filename)
    # print(data["nodes"], data["elems"])
    if not return_elements:
        return data["nodes"]
    else:
        return data["nodes"], data["elems"]

def get_values_from_ens(filename, num_nodes, dim):
    data = read_ens_binary(filename, num_nodes, dim)
    return data["data"]

#=================================#
# grab results
#=================================#

__all__.append('get_case_info')

def get_case_info(casedir):

    basename = get_basefile(casedir) + "_"
    if not os.path.exists(f"{casedir}/results/{basename}mechanical.case"):
        basename = ""
        casefile = os.path.join(casedir, 'results', 'mechanical.case')
        if not os.path.exists(casefile):
            return dict(error="Error preparing simulation.")

    mech_outfile = os.path.join(casedir, f"{basename}mechanical.out")
    with open(mech_outfile) as f:
        if 'Analysis terminated' in f.read():
            return dict(error="Error running simulation.")

    casefile = os.path.join(casedir, 'results', f'{basename}mechanical.case')
    nframes = extract_frames_from_case(casefile)
    if nframes == []:
        return dict(error=f"Case file missing frame count. {casedir}")

    frame_count, semi_frame_count = nframes
    assert (frame_count % 2) == 0, f"Frame count must be even. Got {frame_count}."
    assert (semi_frame_count - 2) == (frame_count - 2) // 2, f"Got incompatible frame counts, {frame_count}, {semi_frame_count}."

    basefilename = f"{casedir}/results/{basename}mechanical00_{frame_count}."
    geo_file = os.path.join(casedir, 'results', f'{basename}mechanical_{semi_frame_count}.geo')
    if not os.path.exists(geo_file):
        return dict(error="Missing Geo file.")

    # # Grab files corresponding to all time-steps
    # NetFabb write state twice for every layer. This is why there are ~twice
    # as many `base_names` as `geo_files`. We want to grab the later file.
    # 
    # For both geo_files and base_names, the last two files correspond to 
    # "cooling" and "substrate removal". We will skip those files for now.
    #
    # Based on mechanical.out, thermal.out

    geo_files  = [] # 0:semi_frame_count
    base_names = [] # 1:frame_count

    for i in range(1, semi_frame_count - 1):
        geo_path  = os.path.join(casedir, 'results', f'{basename}mechanical_{i}.geo')
        assert os.path.exists(geo_path ), f"Geo file {geo_path} not found."
        geo_files.append(geo_path)

        ii = 2 * i
        base_name = os.path.join(casedir, 'results', f'{basename}mechanical00_{ii}')
        dis_path = base_name + '.dis.ens'
        assert os.path.exists(dis_path), f"Base file {dis_path} not found."
        base_names.append(base_name)

    assert len(geo_files) == len(base_names)

    # for i in range(1, semi_frame_count + 1):
    #     geo_path  = os.path.join(casedir, 'results', f'{basename}mechanical_{i}.geo')
    #     assert os.path.exists(geo_path ), f"Geo file {geo_path} not found."
    #     geo_files.append(geo_path)
    #
    # for i in range(frame_count):
    #     ii = i+1
    #     base_name = os.path.join(casedir, 'results', f'{basename}mechanical00_{ii}')
    #     dis_path = base_name + '.dis.ens'
    #     assert os.path.exists(dis_path), f"Base file {dis_path} not found."
    #     base_names.append(base_name)

    return dict(
        geo=geo_file, basefilename=basefilename,
        geo_files=geo_files, base_names=base_names,
    )

def get_finaltime_results(casedir):
    case_info = get_case_info(casedir)
    if len(case_info) == 1:
        return [case_info["error"],] # Case failed
    
    verts, elems = get_vertices_from_geo(case_info["geo"], return_elements=True)
    N_verts = verts.shape[0]

    result_types = dict(dis="disp", ept="strain", rcd="recoater_status", rct="recoater_clearance_percent", sd1="max_dir", sd2="mid_dir", sd3="min_dir",
                        sig="cauchy_stress", sp1="max_stress", sp2="mid_stress", sp3="min_stress", svm="von_mises_stress", tmp="temp")
    result_nums = dict(dis=3, ept=6, rcd=1, rct=1, sd1=3, sd2=3, sd3=3, sig=6, sp1=1, sp2=1, sp3=1, svm=1, tmp=1)

    results = dict(verts=verts, elems=elems)
    for key in result_types:
        result = get_values_from_ens(case_info['basefilename'] + key + ".ens",N_verts, result_nums[key])
        results[result_types[key]] = result.astype(np.float32)

    return results

def get_timeseries_results(casedir):
    case_info = get_case_info(casedir)
    if len(case_info) == 1:
        return [case_info["error"],]

    results = collections.defaultdict(list) # initializes every item to []
    fields  = dict(dis=("disp", 3), svm=("von_mises_stress", 1), tmp=("temp", 1))

    for geo_file in case_info['geo_files']:
        v, e = get_vertices_from_geo(geo_file, return_elements=True)
        v = v.astype(np.float32)
        e = e.astype(np.int32)
        results['verts'].append(v) # [Nv, 3]
        results['elems'].append(e) # [Ne, 8]

    for (i, base_name) in enumerate(case_info['base_names']):
        nv = results['verts'][i].shape[0]
        for key in fields:
            path = base_name + f'.{key}.ens'
            field, dim = fields[key]
            val = get_values_from_ens(path, nv, dim)
            val = val.astype(np.float32)
            results[field].append(val)

    return results

def extract_from_dir(data_dir, out_dir, error_file, timeseries=None):
    os.makedirs(out_dir, exist_ok=True)

    cases = os.listdir(data_dir)
    cases = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    print(f"Loading displacement results from: {data_dir} into {out_dir}")

    if timeseries:
        result_func = get_timeseries_results
    else: # default to final-time
        result_func = get_finaltime_results

    num_success = 0
    num_failure = 0

    with open(error_file,"a") as err:
        for case in tqdm(cases):
            casedir = os.path.join(data_dir, case)
            results = result_func(casedir)
            if len(results) == 1:
                num_failure += 1
                err.write(f'{case}\n')
            else:
                num_success += 1
                if timeseries:
                    out_path = os.path.join(out_dir, case + '.pt')
                    torch.save(results, out_path)
                else:
                    out_path = os.path.join(out_dir, case + '.npz')
                    np.savez(out_path, **results)

    print(f"Successfully saved {num_success} / {num_success + num_failure} cases to NPZ format.")
    print(f"Failed simulation cases are logged to {error_file}")

    return

#=================================#
# assemble data
#=================================#
def unzip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(extract_dir)

def extract_from_zip(source_zip, target_dir, timeseries=None):
    os.makedirs(target_dir, exist_ok=True)

    # extract to temporary directory
    extract_dir = os.path.join(target_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Unzipping {source_zip}...")
    unzip(source_zip, extract_dir)

    # get data
    data_dir = os.path.join(extract_dir, "SandBox")
    err_file = os.path.join(target_dir, "error.txt")
    extract_from_dir(data_dir, target_dir, err_file, timeseries=timeseries)

    # clean up
    print(f"Cleaning up extracted file: {extract_dir}")
    shutil.rmtree(extract_dir)

    return

def extract_zips(source_dir, target_dir, timeseries=None):
    os.makedirs(target_dir, exist_ok=True)
    zip_names = [f for f in os.listdir(source_dir) if f.endswith('.zip')]

    for zip_name in zip_names:
        zip_file = os.path.join(source_dir, zip_name)
        out_dir  = os.path.join(target_dir, zip_name[:-4])

        os.makedirs(out_dir)
        extract_from_zip(zip_file, out_dir, timeseries=timeseries)

    return
#=================================#
#
