import scipy.io as sio
import numpy as np

class taslp_real_RIR_Interface():
	#(n_t60,n_angles,n_mics,rir_len)
	#dist: [1, 2]m
	def __init__(self, dist:int, num_mics):
		self.t60_list = [0.16, 0.36, 0.61]
		self.files_list = [f'aachen_8-8-8-8-8-8-8_roomT60_{t60}.mat' for t60 in self.t60_list] 
		self.scratch_dir = f'/fs/scratch/PAS0774/Shanmukh/Databases/RIRs/taslp_aachen_real_rirs/'
		self.dist = dist
		self.idx_offset = 0 if 1==dist else 13 
		self.rirs_list, self.dp_rirs_list = self.load_all_rirs()
		self.file_num_mics = 8
		self.num_mics = num_mics
	# idx-> degree
	# 0  -> 180 
	def load_all_rirs(self):
		lst = []
		dp_lst = []
		idx_strt, idx_end = 0+self.idx_offset, 13+self.idx_offset # (0-12 (1m), (13-25) (2m))
		for file_name in self.files_list:
			rir = sio.loadmat(f'{self.scratch_dir}{file_name}')
			_h = rir['testingroom']  # testingroom
			x = np.array(_h.tolist()).squeeze()
			x = np.transpose(x,(0,2,1))
			x = x.astype('float32') 
			lst.append(x[idx_strt:idx_end,:,:])
			dp_lst.append(self.get_direct_path_rir(x[idx_strt:idx_end,:,:]))
		
		return lst, dp_lst # list of  arrays with shape (13, 8(n_mics), rir_len) 

	
	def get_direct_path_rir(self, h):
		#h : (26, 8, rir_len)
		fs = 16000
		correction = 1 #int(0.0025*fs)
		h_dp = np.array(h)

		(num_doa, num_ch, rir_len) = h.shape

		idx = np.argmax(np.power(h,2), axis=2)

		for doa_idx in range(num_doa):
			for ch in range(num_ch):
				h_dp[doa_idx, ch, idx[doa_idx, ch]+correction:] = 0

		return h_dp


	def get_rirs(self, t60: float, idx_list: "list integer degrees" ):
		t60_key = self.t60_list.index(t60)
		#rir
		idx_list = [12-idx for idx in idx_list]
		mic_centre_idx = self.file_num_mics//2
		mic_idx_list = self.num_mics//2
		return self.rirs_list[t60_key][idx_list,mic_centre_idx-mic_idx_list:mic_centre_idx+mic_idx_list,:], self.dp_rirs_list[t60_key][idx_list,mic_centre_idx-mic_idx_list:mic_centre_idx+mic_idx_list,:] #(nb_points,  2(n_mics), rir_len)) picking 8cm intermic dist (3:5)



if __name__=="__main__":
	array_type, num_mics, intermic_dist,  room_size = 'linear', 4, 8.0,  ['6', '6', '2.4']
	
	real_rir_interface = taslp_real_RIR_Interface(dist=2, num_mics=num_mics)
	rirs, dp_rirs = real_rir_interface.get_rirs(t60=0.16, idx_list=[idx for idx in range(0,13)])
	breakpoint()
	print(rirs.shape)