from LibriSpeech_utt import *
path = '/fs/scratch/PAS0774/Shanmukh/Databases/LibriSpeech/LibriSpeech/test-clean'
libri_test_dataset = LibriSpeechUttDataset(path, T=1.6, return_vad=True)

from FSDNoisy18k import *

noi_path = '/fs/scratch/PAS0774/Shanmukh/Databases/FSDnoisy18k.audio_test'
T = 1.6
noise_dataset = FSDNoisy18kUttDataset(noi_path, T, return_vad=True)

import yaml
import numpy as np
import pandas as pd
import csv
import random
from data_gen_utils import *
from rir_interface import *
from MovingSpeakerSimulation import *
import h5py

mvng_spk_sim = MovingSpeakerSimulation(16000)

# Test -2S files
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

class CreateTestScenario(object):
	def __init__(self, setup_config, num_mics) -> None:

		self.room_dim_list = setup_config['room_dim']

		self.T60_list = setup_config['t60']
		self.src_mic_dist_list = setup_config['src_mic_dist']
		self.doa_list = [int(_doa) for _doa in range(setup_config['doa_range'][0],setup_config['doa_range'][1]+1, setup_config['doa_resolution'])]

		self.spk_noise_min_angle_gap = setup_config["min_src_noi_gap"]
		self.SIR_list= Parameter(setup_config["SIR"][0], setup_config["SIR"][1])
		self.SNR_list= Parameter(setup_config["SNR"][0], setup_config["SNR"][1])

		self.doa_resolution = setup_config['doa_resolution']

		self.NUM_ITER = 20

		self.num_mics = num_mics

		self.real_rir_interface_1m = taslp_real_RIR_Interface(dist=1, num_mics=num_mics)
		self.real_rir_interface_2m = taslp_real_RIR_Interface(dist=2, num_mics=num_mics)
		

	def gen_scenario_file(self):
		lst = []
		for idx in range(0, self.NUM_ITER):
			for t60 in self.T60_list:
				for src_mic_dist in self.src_mic_dist_list:
					for spk_doa in self.doa_list:
						retry = True
						while(retry):
							noi_doa = random.choice(self.doa_list)
							if noi_doa != spk_doa:
								retry = False
						
						sir = self.SIR_list.getValue(0)
						snr = self.SNR_list.getValue(0)

						spk_idx = np.random.randint(0,len(libri_test_dataset))
						spk_path = libri_test_dataset.file_list[spk_idx]

						noi_idx = np.random.randint(0,len(noise_dataset))
						noi_path = noise_dataset.file_list[noi_idx]

						ex_dict = {
							"spk_path": spk_path,
							"noi_path": noi_path,
							"SIR": sir,
							"SNR": snr,
							"t60" : t60,
							"src_mic_dist": src_mic_dist,
							"spk_doa": spk_doa,
							"noi_doa": noi_doa
						}
		
						lst.append(ex_dict)

		myFile = open('test_files_list.csv', 'w')
		writer = csv.writer(myFile)
		writer.writerow(['spk_path', 'noi_path', 'SIR', 'SNR', 't60', 'src_mic_dist', 'spk_doa', 'noi_doa'])
		for dictionary in lst:
			writer.writerow(dictionary.values())
		myFile.close()

	def read_rirs(self, num_mics, data_frame):
		
		spk_rir_idx = data_frame["spk_doa"]//self.doa_resolution
		noi_rir_idx = data_frame["noi_doa"]//self.doa_resolution

		if data_frame["src_mic_dist"] == 1:
			spk_rirs, dp_rirs = self.real_rir_interface_1m.get_rirs(data_frame["t60"], idx_list=[spk_rir_idx])
			noi_rirs, noi_dp_rirs = self.real_rir_interface_1m.get_rirs(data_frame["t60"], idx_list=[noi_rir_idx])
		else:
			spk_rirs, dp_rirs = self.real_rir_interface_2m.get_rirs(data_frame["t60"], idx_list=[spk_rir_idx])
			noi_rirs, noi_dp_rirs = self.real_rir_interface_2m.get_rirs(data_frame["t60"], idx_list=[noi_rir_idx])
		
		return spk_rirs, noi_rirs, dp_rirs, noi_dp_rirs

	def create_scenario(self, data_frame, seg_len, num_mics):
		spk_rirs, noi_rirs, spk_dp_rirs, noi_dp_rirs = self.read_rirs(num_mics, data_frame)

		spk = data_frame["spk_path"]
		noi = data_frame["noi_path"]

		sph_signal = get_spk_seg(spk, seg_len, is_sph=True) 
		noise_signal = get_spk_seg(noi, seg_len, is_sph = False) 

		radius = data_frame["src_mic_dist"]
		spk_traj_pts = np.expand_dims(np.array([radius*np.cos(np.deg2rad(data_frame["spk_doa"])), radius*np.sin(np.deg2rad(data_frame["spk_doa"])),0]), axis=0)
		noi_traj_pts = np.expand_dims(np.array([radius*np.cos(np.deg2rad(data_frame["noi_doa"])), radius*np.sin(np.deg2rad(data_frame["noi_doa"])),0]), axis=0)

		reverb_sph_signal, src_trajectory_1 = mvng_spk_sim.gen_signal(sph_signal, spk_traj_pts, spk_rirs)
		dp_sph_signal, src_trajectory_1 = mvng_spk_sim.gen_signal(sph_signal, spk_traj_pts, spk_dp_rirs)

		noise_reverb, src_trajectory_2 = mvng_spk_sim.gen_signal(noise_signal, noi_traj_pts, noi_rirs)
		dp_signal_2, src_trajectory_2 = mvng_spk_sim.gen_signal(noise_signal, noi_traj_pts, noi_dp_rirs)

		scale_noi = torch.sqrt(torch.sum(reverb_sph_signal[0,:]**2) / (torch.sum(noise_reverb[0,:]**2) * (10**(data_frame["SIR"]/10)) + 1e-8))

		mix_sph = reverb_sph_signal + scale_noi*noise_reverb

		# Adding microphone noise

		v = torch.normal(0, 1, mix_sph.shape)
		scale_v = torch.sqrt(torch.sum(mix_sph[0,:]**2) / (torch.sum(v[0,:]**2) * (10**(data_frame["SNR"]/10))))

		mix_sph = mix_sph + scale_v*v

		# normalize the root mean square of the mixture to a constant
		sph_len = mix_sph.shape[1]   #*mic_signals.shape[1]          #All mics
		c = 1.0 * torch.sqrt(sph_len / (torch.sum(mix_sph[0,:]**2) + 1e-8))

		mix_sph *= c
		#dp_sph_signal *= c

		_dict = {
		"mix_sph": mix_sph,
		"spk": {
			"dp_signal": dp_sph_signal,
			"doa_mic_axis": data_frame["spk_doa"]
		},
		"noi": {
			"dp_signal": dp_signal_2,
			"doa_mic_axis": data_frame["noi_doa"]
		},
		"SNR": data_frame["SNR"], 
		"t60": data_frame["t60"],
		"SIR": data_frame["SIR"],
		"src_mic_dist": data_frame["src_mic_dist"]
		}
		return _dict




		

if __name__=="__main__":
	config_file = "data_gen.yaml"
	mode = "TestMeasuredRIR"
	with open(config_file,"r") as f:
		exp_config = yaml.safe_load(f)

	df = pd.read_csv("test_files_list.csv")

	#CreateTestScenario(exp_config[mode]).gen_scenario()
	create_scenario = CreateTestScenario(exp_config[mode], 4)

	start_idx = 0

	log_dir_path = f"/fs/scratch/PAA0005/Shanmukh/Habets_SignalAware_Doa/Signals/{mode}/Tr_{start_idx}"
	if not os.path.exists(log_dir_path):
		print(f"Creating Folder rirs_{start_idx}")
		os.makedirs(log_dir_path)

	for idx in range(0, 1561):
		tr_ex = df.iloc[idx]
		mix_dict = create_scenario.create_scenario( tr_ex, int(1.6*16000), 4)

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