###############################################################################
# KEVIN 
###############################################################################
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

###############################################################################
# ANDREW STRESS-STRAIN
###############################################################################

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
    semi_frame_count = frame_count//2+1 #rcd,typ,
    geo_file = f"{filename}/results/{basename}mechanical_{semi_frame_count}.geo"
    displacement_file = f"{filename}/results/{basename}mechanical00_{frame_count}.dis.ens"
    principal_stress_file = f"{filename}/results/{basename}mechanical00_{frame_count}.sd3.ens"
    principal_stress_direction_file = f"{filename}/results/{basename}mechanical00_{frame_count}.sp3.ens"
    cauchy_stress_file = f"{filename}/results/{basename}mechanical00_{frame_count}.sig.ens"
    vonmises_stress_file = f"{filename}/results/{basename}mechanical00_{frame_count}.svm.ens"
    strain_file = f"{filename}/results/{basename}mechanical00_{frame_count}.ept.ens"
    temperature_file = f"{filename}/results/{basename}mechanical00_{frame_count}.tmp.ens"
    recoater_clearance_file = f"{filename}/results/{basename}mechanical00_{frame_count}.rct.ens"
    global_geo = f"{filename}/results/{basename}mechanical_0.geof"
    recoater_clearance_file_global = f"{filename}/results/{basename}mechanical00_{frame_count-1}.grd.ens"
    return dict(geo=geo_file, disp=displacement_file, pstress = principal_stress_file, pstressdir = principal_stress_direction_file, cstress=cauchy_stress_file, vmstress=vonmises_stress_file, strain = strain_file, temp = temperature_file, rc=recoater_clearance_file, ggeo=global_geo, grc=recoater_clearance_file_global,
                frame_count=frame_count, semi_frame_count=semi_frame_count)

def get_displacement_results_only(filename):
    info = get_file_info(filename)
    if len(info) == 1:
        return [info["error"],] # Simulation failed
    verts, elems = get_vertices_from_geo(info["geo"], return_elements=True)
    N_verts = verts.shape[0]
    disp = get_values_from_ens(info["disp"], N_verts, 3)
    principal_stress = get_values_from_ens(info["pstress"], N_verts, 3)
    principal_stress_dir =  get_values_from_ens(info["pstressdir"], N_verts, 1)
    cauchy_stress = get_values_from_ens(info["cstress"], N_verts, 6)
    vonmises_stress = get_values_from_ens(info["vmstress"], N_verts, 1)
    strain = get_values_from_ens(info["strain"], N_verts, 3)
    temp = get_values_from_ens(info["temp"], N_verts, 1)
    return verts, elems, disp, principal_stress, principal_stress_dir, cauchy_stress, vonmises_stress, strain, temp

def extract_data(full_dataset, new_dataset, error_file):
    if not os.path.isdir(new_dataset):
        os.mkdir(new_dataset)
    names = os.listdir(full_dataset)
    num_files = len(names)
    num_success = 0
    print(f"Loading displacement results from: {full_dataset} into {new_dataset}")
    with open(error_file,"w") as err:
        for i, name in enumerate(names):
            results = get_displacement_results_only(os.path.join(full_dataset, name))
            if len(results) == 1:
                err.write(f'{name}\n')
            else:
                num_success += 1
                verts, elems, disp, pstress, pstressdir, cstress, vmstress, strain, temp = results
                #Save files with .TYPE in new output file
                disp_output_path = os.path.join(new_dataset, name + '_disp.npz')
                np.savez(disp_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), disp=disp.astype(np.float32))
                principal_stress_output_path = os.path.join(new_dataset, name + '_pstress.npz')
                np.savez(principal_stress_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), pstress=pstress.astype(np.float32))
                principal_stress_dir_output_path = os.path.join(new_dataset, name + '_pstressdir.npz')
                np.savez(principal_stress_dir_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), pstressdir=pstressdir.astype(np.float32))
                cauchy_stress_output_path = os.path.join(new_dataset, name + '_cstress.npz')
                np.savez(cauchy_stress_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), cstress=cstress.astype(np.float32))
                vonmises_stress_output_path = os.path.join(new_dataset, name + '_vmstress.npz')
                np.savez(vonmises_stress_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), vmstress=vmstress.astype(np.float32))
                strain_output_path = os.path.join(new_dataset, name + '_strain.npz')
                np.savez(strain_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), strain=strain.astype(np.float32))
                temp_output_path = os.path.join(new_dataset, name + '_temp.npz')
                np.savez(temp_output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), temp=temp.astype(np.float32))
    print(f"Done. {num_success} of {num_files} simulations successful.")
    print(f"Failed simulation file names are logged to {error_file}")

###############################################################################
# ANDREW DISPLACEMENT
###############################################################################
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
    semi_frame_count = frame_count//2+1
    geo_file = f"{filename}/results/{basename}mechanical_{semi_frame_count}.geo"
    displacement_file = f"{filename}/results/{basename}mechanical00_{frame_count}.dis.ens"
    recoater_clearance_file = f"{filename}/results/{basename}mechanical00_{frame_count}.rct.ens"
    global_geo = f"{filename}/results/{basename}mechanical_0.geof"
    recoater_clearance_file_global = f"{filename}/results/{basename}mechanical00_{frame_count-1}.grd.ens"

    return dict(geo=geo_file, disp=displacement_file, rc=recoater_clearance_file, ggeo=global_geo, grc=recoater_clearance_file_global,
                frame_count=frame_count, semi_frame_count=semi_frame_count)

def get_displacement_results_only(filename):
    info = get_file_info(filename)
    if len(info) == 1:
        return [info["error"],] # Simulation failed
    verts, elems = get_vertices_from_geo(info["geo"], return_elements=True)
    N_verts = verts.shape[0]
    disp = get_values_from_ens(info["disp"], N_verts, 3)
    return verts, elems, disp

def extract_data(full_dataset, new_dataset, error_file):
    if not os.path.isdir(new_dataset):
        os.mkdir(new_dataset)
    names = os.listdir(full_dataset)
    num_files = len(names)
    num_success = 0
    print(f"Loading displacement results from: {full_dataset} into {new_dataset}")
    with open(error_file,"w") as err:
        for i, name in enumerate(names):
            print_status_bar(i,num_files)
            results = get_displacement_results_only(os.path.join(full_dataset, name))
            if len(results) == 1:
                err.write(f'{name}\n')
            else:
                num_success += 1
                verts, elems, disp = results
                output_path = os.path.join(new_dataset, name + '.npz')
                np.savez(output_path, verts=verts.astype(np.float32), elems=elems.astype(np.int32), disp=disp.astype(np.float32))
                print(f"{name} elems = {elems}")
    print(f"Done. {num_success} of {num_files} simulations successful.             ")
    print(f"Failed simulation file names are logged to {error_file}")


