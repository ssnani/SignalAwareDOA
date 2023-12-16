import numpy as np
import h5py
import torch
import torchaudio
import pandas as pd
import sys
import webrtcvad
import os

sys.path.insert(1, '../Code/')
np.random.seed(0)

# MovingSpeakerSimualtion
from MovingSpeakerSimulation import *

mvng_spk_sim = MovingSpeakerSimulation(16000)

def _cleanSilences(s, aggressiveness, return_vad=False, fs=16000):
	vad = webrtcvad.Vad()
	vad.set_mode(aggressiveness)

	vad_out = np.zeros_like(s)
	vad_frame_len = int(10e-3 * fs)
	n_vad_frames = len(s) // vad_frame_len
	for frame_idx in range(n_vad_frames):
		frame = s[:,frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
		frame_bytes = (frame * 32767).astype('int16').tobytes()
		vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = vad.is_speech(frame_bytes, fs)
	s_clean = s * vad_out

	return (s_clean, vad_out) if return_vad else s_clean

def get_spk_seg(sph_utt_path, req_samples, is_sph=True):

	retry = True
	while retry:
		utterance, fs = torchaudio.load(sph_utt_path)
		#if is_sph:
		#	utterance = _cleanSilences(utterance.numpy(), 3)
		#	utterance = torch.from_numpy(utterance)

		if 16000!= fs:
			utterance = torchaudio.functional.resample(utterance,fs, 16000)
			fs = 16000

		#energy check
		if torch.sum(utterance**2)==0:
			utterance = torch.randn(1, 64000)
			print("Generating Gaussion random noise signal")

		if 16000==fs:
			utt_len = utterance.shape[1]
			if utt_len > req_samples:
				start_idx = np.random.randint(0, utt_len - req_samples)
				utterance = utterance[:, start_idx:start_idx+ req_samples]
			else:
				#zero pad
				extra_samples = req_samples - utt_len
				if is_sph:
					utterance = torch.cat([utterance, torch.zeros(1,extra_samples)], dim=1)
				else:
					req_repetitions = int(np.ceil(req_samples/utt_len))
					utterance = utterance.repeat(1, req_repetitions)
					utterance = utterance[:,:req_samples]
			
			utterance -= utterance.mean()
			
		retry = True if torch.sum(utterance**2)==0 else False
		norm_fn = lambda x: x * torch.sqrt(x.shape[1] / (torch.sum(x**2) + 1e-8))
		signal_seg = norm_fn(utterance) 

	return signal_seg

def read_rirs_h5(rir_path):  
		
	room_cfg = {}
			 
	with h5py.File(rir_path, 'r') as f:

		for k, v in f["room_cfg"].items():
			room_cfg[k] = np.array(v)

		
		spk_info, noi_info = {}, {}
		for k, v in f[f"spk_info_room"].items():
			spk_info[k] = np.array(v)

		for k, v in f[f"noi_info_room"].items():
			noi_info[k] = np.array(v)
			
	return room_cfg, spk_info, noi_info

def compute_azimuth(centre_mic_pos, mic_rotation, spk_traj_pts):
	x_cord_centre_mic, y_cord_centre_mic = centre_mic_pos[0], centre_mic_pos[1]

	x_cord_spk, y_cord_spk = spk_traj_pts[:,0], spk_traj_pts[:,1]

	return ((np.rad2deg(np.arctan2((y_cord_spk - y_cord_centre_mic), (x_cord_spk - x_cord_centre_mic))))%360  - (mic_rotation)%360)%360

def create_mix(data_frame, seg_len):

	room_cfg, spk_info, noi_info = read_rirs_h5(data_frame["rirs"])

	spk = data_frame["spk"]
	noi = data_frame["noi"]

	spk_traj_pts, spk_rirs, spk_dp_rirs = spk_info["spk_traj_pts"], spk_info["rirs"], spk_info["dp_rirs"]
	noi_traj_pts, noi_rirs, noi_dp_rirs = noi_info["noi_traj_pts"], noi_info["rirs"], noi_info["dp_rirs"]

	# segment level 
	#sph_signal = get_spk_seg(spk, seg_len, is_sph=True)   
	#noise_signal = get_spk_seg(noi, seg_len, is_sph = False) 

	#File level


	#scaling
	SNR = data_frame["snr"]
	
	reverb_sph_signal, src_trajectory_1 = mvng_spk_sim.gen_signal(sph_signal, spk_traj_pts, spk_rirs)
	dp_sph_signal, src_trajectory_1 = mvng_spk_sim.gen_signal(sph_signal, spk_traj_pts, spk_dp_rirs)

	noise_reverb, src_trajectory_2 = mvng_spk_sim.gen_signal(noise_signal, noi_traj_pts, noi_rirs)
	dp_signal_2, src_trajectory_2 = mvng_spk_sim.gen_signal(noise_signal, noi_traj_pts, noi_dp_rirs)


	scale_noi = torch.sqrt(torch.sum(reverb_sph_signal[0,:]**2) / (torch.sum(noise_reverb[0,:]**2) * (10**(SNR/10)) + 1e-8))

	mix_sph = reverb_sph_signal + scale_noi*noise_reverb

	# Adding microphone noise

	v = torch.normal(0, 1, mix_sph.shape)

	SNR_v = -10 * torch.rand(1) + 30

	scale_v = torch.sqrt(torch.sum(mix_sph[0,:]**2) / (torch.sum(v[0,:]**2) * (10**(SNR_v/10))))

	mix_sph = mix_sph + scale_v*v

	# normalize the root mean square of the mixture to a constant
	sph_len = mix_sph.shape[1]   #*mic_signals.shape[1]          #All mics
	c = 1.0 * torch.sqrt(sph_len / (torch.sum(mix_sph[0,:]**2) + 1e-8))

	mix_sph *= c
	dp_sph_signal *= c
	#dp_signal_2 *= c

	# w.r.t microphone axis (0, 1) i.e local reference
	spk_doa = spk_info["spk_doa_mic_axis"]
	noi_doa = noi_info["noi_doa_mic_axis"]

	_dict = {
		"mix_sph": mix_sph,
		"spk": {
			"dp_signal": dp_sph_signal,
			"doa_mic_axis": spk_doa
		},
		"noi": {
			"dp_signal": dp_signal_2,
			"doa_mic_axis": noi_doa
		},
		"SNR_v": SNR_v, 
		"t60": room_cfg["t60"],
		"SNR": SNR
	}
	return _dict
	


if __name__ == "__main__":
	start_idx = int(sys.argv[1]) -1
	df =  pd.read_csv("../Code/validation_files_list.csv")
	mode ="Validation_correction"

	NUM_FILES_PER_FOLDER = 5000
	start_idx *= NUM_FILES_PER_FOLDER
	log_dir_path = f"/fs/scratch/PAA0005/Shanmukh/Habets_SignalAware_Doa/Signals/{mode}/Tr_{start_idx}"
	if not os.path.exists(log_dir_path):
		print(f"Creating Folder rirs_{start_idx}")
		os.makedirs(log_dir_path)

	for idx in range(start_idx, start_idx+NUM_FILES_PER_FOLDER):
		tr_ex = df.iloc[idx]
		mix_dict = create_mix(tr_ex, seg_len=int(1.6*16000))

		with h5py.File(f"{log_dir_path}/tr_ex_{idx}.h5", 'w') as f:
			f.create_dataset("mix_sph", data = mix_dict["mix_sph"].numpy())

			spk = f.create_group("spk")
			noi = f.create_group("noi")

			spk["dp_signal"] = mix_dict["spk"]["dp_signal"].numpy()
			spk["doa_mic_axis"] = mix_dict["spk"]["doa_mic_axis"]

			noi["dp_signal"] = mix_dict["noi"]["dp_signal"].numpy()
			noi["doa_mic_axis"] = mix_dict["noi"]["doa_mic_axis"]

			# for debug purposes (plots)
			ex_gen_cfg = f.create_group("ex_gen_cfg")
			for k, v in tr_ex.items():
				ex_gen_cfg[k] = v
			








	



