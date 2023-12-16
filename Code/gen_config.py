#The file deals with generating configuration files that mimics the setup mentioned in the paper

# **** Here random configurations are picked/chosen from a SET { } of pre-defined values.


import yaml
import numpy as np
import random
from data_gen_utils import *
from array_setup import *

class CreateScenario(object):
	def __init__(self, mic_array, setup_config) -> None:

		self.mic_array = mic_array
		self.room_dim_list = ParameterSet(setup_config['room_dim'])

		self.T60_list = ParameterSet(setup_config['t60'])
		self.src_mic_dist_list = ParameterSet(setup_config['src_mic_dist'])
		self.doa_list = ParameterSet([int(_doa) for _doa in range(setup_config['doa_range'][0],setup_config['doa_range'][1]+1, setup_config['doa_resolution'])])

		self.mic_pos_from_room_centre = Parameter([-0.5, -0.5, 0], [0.5, 0.5, 0]) # metres (considering upto one decimal precision)
		self.mic_rotation_range = Parameter(-60, 60)         # degrees (considering upto zero decimal precision)
		self.spk_noise_min_angle_gap = setup_config["min_src_noi_gap"]

	
	def gen_linear_array_with_rotation(self, centre_mic_pos, mic_array: "np.array(num_mics, 3)", rotation: "degrees"):
		num_mics = mic_array.shape[0]
		rotated_mic_array = np.zeros((num_mics,3)) # w.r.t room
		for mic_idx in range(0, num_mics):
			radius = mic_array[mic_idx,0]  # assuming mic_array elements on x-axis ( x_cord, 0, 0)
			rotated_mic_array[mic_idx] = np.array([centre_mic_pos[0] + radius*np.cos(np.deg2rad(rotation)), centre_mic_pos[1] + radius*np.sin(np.deg2rad(rotation)), centre_mic_pos[2]])
		return rotated_mic_array
	
	def gen_room_config(self):
		
		#step:1 select room size
		self.room_size = np.array(self.room_dim_list.getValue())       #shape (3, )
		#step:2 select t60 
		self.t60 = self.T60_list.getValue()

		#step3: select microphone array pos
		self.room_centre = np.round(self.room_size/2, 1)	
		self.mic_centre_pos = self.room_centre + self.mic_pos_from_room_centre.getValue(round_format=1)
		self.mic_rotation = self.mic_rotation_range.getValue(round_format=0)
		self.mic_array_pos = self.gen_linear_array_with_rotation(self.mic_centre_pos, self.mic_array, self.mic_rotation)   #rotated mic array indices in the room
		
		config_dict = {
			"room_size"         : self.room_size,
			"t60"               : self.t60,
			"mic_centre_pos"    : self.mic_centre_pos,
			"mic_rotation"      : self.mic_rotation,      # w.r.t x-axis assuming mic_centre as origin
			"mic_array_pos"     : self.mic_array_pos
		}

		return config_dict

	# speaker is stationary
	def speaker_pos(self, room_config_dict):
		retry = True
		while(retry):
			src_mic_dist = self.src_mic_dist_list.getValue()

			spk_doa = self.doa_list.getValue() # w.r.t mic_array axis , *** This is the label for doa prediction
			
			#relative spk_doa in the room because of mic rotation w.r.t X-axis
			rel_spk_doa = room_config_dict['mic_rotation'] + spk_doa 
			centre_mic_pos = room_config_dict['mic_centre_pos']
			spk_traj_pts = np.expand_dims(np.array([centre_mic_pos[0] + src_mic_dist*np.cos(np.deg2rad(rel_spk_doa)), centre_mic_pos[1] + src_mic_dist*np.sin(np.deg2rad(rel_spk_doa)), centre_mic_pos[2] + 0]), axis = 0)

			is_outside_room = ((spk_traj_pts<=0).any()) or ((spk_traj_pts >= room_config_dict["room_size"]).any())

			if not is_outside_room:
				retry = False


		spk_conifg_dict = {
			"spk_mic_dist": src_mic_dist,
			"spk_traj_pts": spk_traj_pts,   # (nb_points, 3(x,y,z))
			"spk_doa_mic_axis": spk_doa
		}

		return spk_conifg_dict

	def noise_pos(self, room_config_dict, spk_conifg_dict):
		noi_mic_dist = spk_conifg_dict['spk_mic_dist']
		spk_doa = spk_conifg_dict["spk_doa_mic_axis"]

		retry=True
		while(retry):
			retry_doa=True
			while(retry_doa):
				noi_doa = self.doa_list.getValue()
				if np.abs(spk_doa - noi_doa) > self.spk_noise_min_angle_gap:
					retry_doa = False
			
			rel_noi_doa = room_config_dict['mic_rotation'] + noi_doa

			centre_mic_pos = room_config_dict['mic_centre_pos']
			noi_traj_pts = np.expand_dims(np.array([centre_mic_pos[0] + noi_mic_dist*np.cos(np.deg2rad(rel_noi_doa)), centre_mic_pos[1] + noi_mic_dist*np.sin(np.deg2rad(rel_noi_doa)), centre_mic_pos[2] + 0]), axis = 0)

			is_outside_room = ((noi_traj_pts<=0).any()) or ((noi_traj_pts >= room_config_dict["room_size"]).any())

			if not is_outside_room:
				retry = False


		noi_conifg_dict = {
			"noi_mic_dist": noi_mic_dist,
			"noi_traj_pts": noi_traj_pts,   # (nb_points, 3(x,y,z))
			"noi_doa_mic_axis": noi_doa
		}

		return noi_conifg_dict

if __name__=="__main__":
	config_file = "data_gen.yaml"
	mode = "Train"
	with open(config_file,"r") as f:
		exp_config = yaml.safe_load(f)

	mic_array = get_array_set_up_from_config("linear", 4, 8)
	
	acoustic_scene = CreateScenario(mic_array.mic_pos, exp_config[mode])

	config_with_mic_array = acoustic_scene.gen_room_config()

	print(config_with_mic_array)

	spk_conifg_dict = acoustic_scene.speaker_pos(config_with_mic_array)

	print(spk_conifg_dict)

	noi_conifg_dict = acoustic_scene.noise_pos(config_with_mic_array, spk_conifg_dict)

	print(noi_conifg_dict)
	breakpoint()









		
		
		


