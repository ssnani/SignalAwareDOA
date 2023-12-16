import torch
import numpy as np
import math


class SignalAwareDoA_features(object):
	def __init__(self, frame_len, frame_shift, doa_resolution = 5, array_type=None, array_setup=None, fs=16000):
		super().__init__()

		self.frame_len = frame_len
		self.frame_shift = frame_shift
		self.kernel = (1,frame_len)
		self.stride = (1,frame_shift)
		#self.mic_idx = ref_mic_idx   
		self.array_type = array_type
		self.array_setup = array_setup

		self.freq_range = self.frame_len//2 + 1

		self.doa_resolution = doa_resolution #degrees


	def __call__(self, mix_signal, tgt, DOA):  
		epsilon = 10**-10
		mix_cs = torch.stft(mix_signal, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=True) #(2, num_freq, T)

		#For ADNN
		mix_mag_0 = torch.abs(mix_cs[0,:,:])
		mix_mag_0 = torch.permute(mix_mag_0, [1,0])

		# for DDNN
		mix_ph = torch.angle(mix_cs)
		num_channels, num_freq, num_frames = mix_ph.shape
		mix_ph = torch.permute(mix_ph, [2,0,1])

		tgt_cs = torch.stft(tgt, self.frame_len, self.frame_shift, self.frame_len, torch.hamming_window(self.frame_len), center=False, return_complex=True)
		tgt_ph = torch.angle(tgt_cs)
		tgt_ph = torch.permute(tgt_ph, [2,0,1])

		doa_cls = DOA//5
		return mix_mag_0, mix_ph, tgt_cs, doa_cls.to(torch.long)