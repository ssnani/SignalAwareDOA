import os
import glob
import re
import torch
from network_input_output import *
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class SignalAwareDoADataset(Dataset):
	def __init__(self, root_path, transforms, size=None):

		self.root_path = root_path
		self.files_list = glob.glob(root_path+"/*/*.h5")

		self.files_list = sorted(self.files_list, key=lambda s: int(re.search(r'\d+', s[-8:]).group()))
		self.transforms = transforms
		self.size = size

	def __len__(self):
		return len(self.files_list) if self.size is None else self.size

	def __getitem__(self, idx):

		ex_file = self.files_list[idx]

		with h5py.File(ex_file, 'r') as f:
			mix_sph = torch.from_numpy(np.array(f["mix_sph"]))
			spk_dp_signal = torch.from_numpy(np.array(f["spk"]["dp_signal"]))

			spk_doa = torch.from_numpy(np.array(f["spk"]["doa_mic_axis"]))
			noi_doa = torch.from_numpy(np.array(f["noi"]["doa_mic_axis"]))

		if self.transforms is not None:
			for t in self.transforms:
				mix_mag, mix_ph, tgt_spk_cs, spk_doa = t(mix_sph, spk_dp_signal, spk_doa)

			return mix_mag, mix_ph, tgt_spk_cs, spk_doa
		else:
			return mix_sph, spk_dp_signal, spk_doa, noi_doa
		

if __name__=="__main__":

	root_path = "/fs/scratch/PAA0005/Shanmukh/Habets_SignalAware_Doa/Signals/Train/"
	transforms = [SignalAwareDoA_features(frame_len=512, frame_shift=256, doa_resolution = 5, array_type=None, array_setup=None, fs=16000)]
	train_dataset = SignalAwareDoADataset(root_path, transforms)

	train_dl = DataLoader(train_dataset, batch_size=16, num_workers=5)
	#breakpoint()
	for idx, ex in enumerate(train_dl):
		mix_sph_mag, mix_sph_ph, tgt_spk_cs, spk_doas = ex
		print(f"{mix_sph_mag.shape}, {mix_sph_ph.shape}, {tgt_spk_cs.shape}, {spk_doas}")
		#print(f"{mix_sph_mag.dtype}, {mix_sph_ph.dtype}, {tgt_spk_cs.dtype}, {spk_doas.dtype}")
		if torch.isnan(mix_sph_mag).any():
			print(f"idx: {idx}, {torch.where(torch.isnan(mix_sph_mag))}")
		#break