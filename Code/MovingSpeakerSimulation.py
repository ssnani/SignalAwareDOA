import numpy as np
import torch
import torchaudio
from scipy.signal import fftconvolve

GPU_RIR_IMPLEMENTATION = False

class MovingSpeakerSimulation(object):
	def __init__(self, fs) -> None:
		self.fs = 16000

	def get_timestamps(self, traj_pts, sig_len):
		nb_points = traj_pts.shape[0]
		
		# Interpolate trajectory points
		timestamps = np.arange(nb_points) * sig_len / self.fs / nb_points
		t = np.arange(sig_len)/self.fs
		trajectory = np.array([np.interp(t, timestamps, traj_pts[:,i]) for i in range(3)]).transpose()

		return timestamps, t, trajectory
	
	def simulate_source(self, signal, RIRs, timestamps, t):
		# signal: tensor( 1, sig_len), 
		# RIRs: numpy (nb_points, num_ch, rir_len)
		# reverb_signals : tensor( num_ch, sig_len)

		
		reverb_signals = self.simulateTrajectory(signal, RIRs, timestamps=timestamps, fs=self.fs) #torch.from_numpy(signal).unsqueeze(dim=0)
		reverb_signals = reverb_signals[:,0:len(t)]
		return reverb_signals
	
	def fftconv(self, signal, RIR):
		# Numpy implementation
		# signal: ( 1, sig_len), rir: (num_ch, rir_len)
		
		reverb_signal = fftconvolve(signal, RIR, mode='full')
		return reverb_signal

	def get_seg(self, signal, timestamps):
		blk_len = signal.shape[-1] if len(timestamps)==1 else int(timestamps[1]*self.fs)
		seg_sig = torch.nn.functional.unfold(signal.unsqueeze(dim=1).unsqueeze(dim=1), kernel_size=(1, blk_len), padding=(0,0), stride=(1, blk_len))
		seg_sig = torch.permute(seg_sig.squeeze(dim=0),[1,0])   #(num_seg, blk_len)

		return seg_sig

	def simulateTrajectory(self, signal, RIRs, timestamps, fs):
		#signal: tensor( 1, sig_len)
		#RIRs: numpy (nb_points, num_ch, rir_len)
		
		(nb_points, num_ch, rir_len) = RIRs.shape
		nSamples = signal.shape[-1]
		w_ini = np.append((timestamps*fs).astype(int), nSamples)

		seg_signal = self.get_seg(signal, timestamps)

		reverb_signal = torch.zeros(num_ch, nSamples+rir_len-1)

		for seg_idx in range(nb_points):
			#reverb_seg = self.conv(seg_signal[[seg_idx],:], RIRs[seg_idx,:,:])
			
			reverb_seg = torch.from_numpy(self.fftconv(seg_signal[[seg_idx],:].numpy(), RIRs[seg_idx,:,:]))

			reverb_signal[:,w_ini[seg_idx] : w_ini[seg_idx+1]+rir_len-1] += reverb_seg

		return reverb_signal
	
	def gen_signal(self, signal: "torch tensor", src_traj_pts: "[nb_points, 3]", rirs: "numpy array [nb_points, num_ch, rir_len]"):
		src_timestamps, t, src_trajectory = self.get_timestamps(src_traj_pts, signal.shape[1])
		if GPU_RIR_IMPLEMENTATION:
			reverb_sig = self.simulate_source_gpuRIR(signal[0].numpy(), rirs, src_timestamps, t)
		else:
			reverb_sig = self.simulate_source(signal, rirs, src_timestamps, t)

		return reverb_sig, src_trajectory
	


	def rms_normalize(self, mic_signals, dp_signals):
		# normalize the root mean square of the mixture to a constant
		sph_len = mic_signals.shape[1]   #*mic_signals.shape[1]          #All mics

		c = 1.0 * torch.sqrt(sph_len / (torch.sum(mic_signals[0,:]**2) + 1e-8))

		mic_signals *= c
		dp_signals *= c
		return mic_signals, dp_signals