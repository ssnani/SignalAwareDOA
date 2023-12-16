import numpy as np
import sys
sys.path.insert(1, '../Code/')
from gen_config import  *

import gpuRIR
from collections import namedtuple

import h5py
import sys
import os

from array_setup import *

array_setup = get_array_set_up_from_config("linear", 4, 8)

def create_rirs(config_with_mic_array, T60, room_spk_traj_pts: "[nb_points, 3]", fs):

	room_size = config_with_mic_array["room_size"]
	#T60 = config_with_mic_array["t60"]
	room_mic_pos = config_with_mic_array["mic_array_pos"]
	#room_spk_traj_pts = spk_config_dict["spk_traj_pts"]

	abs_weights = [1]*6
	beta = gpuRIR.beta_SabineEstimation(room_size, T60, abs_weights)

	if T60 == 0:
		Tdiff = 0.1
		Tmax = 0.1
		nb_img = [1,1,1]
	else:
		if 0:
		#original code using ISM at start and replace with diffuse model after certain time
			Tdiff = gpuRIR.att2t_SabineEstimator(12, T60) # Use ISM until the RIRs decay 12dB
			Tmax = gpuRIR.att2t_SabineEstimator(40, T60)  # Use diffuse model until the RIRs decay 40dB
		
		#Using only ISM
		Tmax = gpuRIR.att2t_SabineEstimator(60, T60)  # Use diffuse model until the RIRs decay 60dB
		Tdiff = Tmax

		if T60 < 0.15: Tdiff = Tmax # Avoid issues with too short RIRs
		nb_img = gpuRIR.t2n( Tdiff, room_size )

	nb_mics  = len(room_mic_pos)
	nb_traj_pts = len(room_spk_traj_pts)
	nb_gpu_calls = min(int(np.ceil( fs * Tdiff * nb_mics * nb_traj_pts * np.prod(nb_img) / 1e9 )), nb_traj_pts)
	traj_pts_batch = np.ceil( nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls+1) ).astype(int)

	RIRs_list = [ ]
	for i in range(0,nb_gpu_calls):
		RIRs_list.append( gpuRIR.simulateRIR(room_size, beta,
						room_spk_traj_pts[traj_pts_batch[i]:traj_pts_batch[i+1],:], room_mic_pos,
						nb_img, Tmax, fs, Tdiff=Tdiff,
						orV_rcv=array_setup.mic_orV, mic_pattern='omni') )
	RIRs = np.concatenate(RIRs_list, axis=0)

	return RIRs

if __name__=="__main__":
	sys.path.insert(1, '../Code/')
	config_file = "../Code/data_gen.yaml"
	mode = "Validation"
	with open(config_file,"r") as f:
		exp_config = yaml.safe_load(f)
	
	acoustic_scene = CreateScenario(array_setup.mic_pos, exp_config[mode])

	#print(sys.argv)
	start_idx = int(sys.argv[1])-1
	#print(start_idx)
	NUM_FILES_PER_FOLDER = 5000
	start_idx *= NUM_FILES_PER_FOLDER 

	num_src = 2
			 
	log_dir_path = f"/fs/scratch/PAA0005/Shanmukh/Habets_SignalAware_Doa/RIRs/{mode}/Rirs_{start_idx}"
	if not os.path.exists(log_dir_path):
		print(f"Creating Folder rirs_{start_idx}")
		os.makedirs(log_dir_path)

	for _idx in range(start_idx, start_idx+NUM_FILES_PER_FOLDER):

		room_config_with_mic_array = acoustic_scene.gen_room_config()
		with h5py.File(f"{log_dir_path}/rirs_{_idx}.h5", 'w') as f:
			room_cfg = f.create_group("room_cfg")
			for k,v in room_config_with_mic_array.items():
				room_cfg[k] = v

			# spk 
			spk_config_dict = acoustic_scene.speaker_pos(room_config_with_mic_array)

			rirs = create_rirs(room_config_with_mic_array, room_config_with_mic_array["t60"], spk_config_dict["spk_traj_pts"], 16000)
			dp_rirs = create_rirs(room_config_with_mic_array, 0, spk_config_dict["spk_traj_pts"], 16000)

			spk_config_dict["traj_shape"] = "stationary"
			spk_config_dict["rirs"] = rirs
			spk_config_dict["dp_rirs"] = dp_rirs

			spk_info = f.create_group(f"spk_info_room")

			for k,v in spk_config_dict.items():
				spk_info[k] = v

			#noi
			noi_config_dict = acoustic_scene.noise_pos(room_config_with_mic_array, spk_config_dict)
			noi_rirs = create_rirs(room_config_with_mic_array, room_config_with_mic_array["t60"], noi_config_dict["noi_traj_pts"], 16000)
			noi_dp_rirs = create_rirs(room_config_with_mic_array, 0, noi_config_dict["noi_traj_pts"], 16000)

			noi_config_dict["traj_shape"] = "stationary"
			noi_config_dict["rirs"] = noi_rirs
			noi_config_dict["dp_rirs"] = noi_dp_rirs

			noi_info = f.create_group(f"noi_info_room")

			for k,v in noi_config_dict.items():
				noi_info[k] = v






