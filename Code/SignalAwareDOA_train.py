import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary, LearningRateMonitor

import os
import sys 
import random

from dataset import SignalAwareDoADataset
from network_input_output import SignalAwareDoA_features
from network import SignalAwareDoA

#from direct_approach_callback import Direct_approach_doa_callback
from array_setup import get_array_set_up_from_config
from arg_parser import parser
from debug import dbg_print
import matplotlib.pyplot as plt
import math
import numpy as np
from callbacks import *

class SignalAwareDoA_model(pl.LightningModule):
	def __init__(self, bidirectional: bool, num_mics: int, num_freq: int, num_doa_classes: int, train_dataset: Dataset, val_dataset: Dataset, batch_size=32, num_workers=4):
		super().__init__()
		pl.seed_everything(77)

		self.model = SignalAwareDoA(hidden_size=512, bidirectional=False, num_mics=num_mics, num_freq=num_freq, num_doa_classes=num_doa_classes)

		self.loss_doa = nn.BCELoss() #nn.CrossEntropyLoss(ignore_index=-1) 

		self.batch_size = batch_size
		self.num_workers = num_workers
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def val_dataloader(self):		
		return DataLoader(self.val_dataset, batch_size = self.batch_size, 
							 num_workers=self.num_workers, pin_memory=True, drop_last=True)

	def configure_optimizers(self):
		_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, amsgrad=True)   #TODO:
		_lr_scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=2, gamma=0.98)
		#_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer, factor=0.5, patience=3)
		return {"optimizer": _optimizer, "lr_scheduler": _lr_scheduler, "monitor": 'val_loss'}

	def forward_step(self, input_batch, mode_str: str, batch_idx):
		mix_mag, mix_ph, tgt_spk_cs, doa_indices = input_batch

		est_doa_logits = self.model(mix_mag, mix_ph)

		batch_size, num_frms, num_doa_classes = est_doa_logits.shape
		_est_doa_logits = torch.reshape(est_doa_logits, (batch_size*num_frms, num_doa_classes))

		#averaging across time
		est_doa_logits_utt = torch.mean(est_doa_logits, dim=1)
		_doa_indices = torch.reshape(doa_indices, (-1,))

		one_hot_doa = F.one_hot(_doa_indices, 37).to(torch.float)
		
		loss_doa = self.loss_doa(est_doa_logits_utt, one_hot_doa)

		#print(loss_doa)

		#grad_norm, grad_data_ratio = gradient_norm_per_layer(self.model)

		#print(f"batch_idx: {batch_idx}, grad_norm: {grad_norm}")

		self.log(f'{mode_str}_loss', loss_doa, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True )	

		return {"loss": loss_doa, "est_doa_prob": est_doa_logits, "doa_indices": doa_indices } 
		
	def training_step(self, train_batch, batch_idx):
		loss_est_dict = self.forward_step(train_batch, 'train', batch_idx)
		return loss_est_dict

	def validation_step(self, val_batch, batch_idx):
		loss_est_dict = self.forward_step(val_batch, 'val', batch_idx)
		return loss_est_dict

	def test_step(self, test_batch, batch_idx):
		loss_est_dict = self.forward_step(test_batch, 'test', batch_idx)
		return loss_est_dict

	def validation_epoch_end(self, validation_step_outputs):
		tensorboard = self.logger.experiment
		softmax = nn.Softmax(dim=1)
		if (self.current_epoch % 2==0):
			for batch_idx, batch_output in enumerate(validation_step_outputs):
				if ((batch_idx+self.current_epoch)%4==0):
					idx = random.randint(0, self.batch_size-1)
					est_doa_prob = batch_output['est_doa_prob'][idx]
					#doa_prob = softmax(est_doa_logits)
					est_doa_idx = torch.argmax(est_doa_prob, dim=1)

					doa_indices = batch_output['doa_indices'][idx]
					#print(batch_idx, idx, doa_indices, est_doa_idx)
					fig = plt.figure()		
					plt.plot(np.repeat(doa_indices.cpu().numpy(), 99), label="tgt_doa")
					plt.plot(est_doa_idx.cpu().numpy(), '*', label="est_doa")
					plt.legend()
					tensorboard.add_figure(f'fig_ep{self.current_epoch}_b{batch_idx}_i{idx}', fig)

def main(args):
	dbg_print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
	T = 1.6
	#array jobs code changes
	#reading from file for array jobs
	if args.array_job:

		T60 = None
		SNR = None
		with open(args.input_train_filename, 'r') as f:
			lines = [line for line in f.readlines()]

		array_config = {}
		#key value 
		for line in lines:
			lst = line.strip().split()
			if lst[0]=="dataset_dtype":
				dataset_dtype = lst[1]
			elif lst[0]=="dataset_condition":
				dataset_condition = lst[1]
			elif lst[0]=="noise_simulation":
				noise_simulation = lst[1]
			elif lst[0]=="ref_mic_idx":
				ref_mic_idx = int(lst[1])
			elif lst[0]=="dataset_file":
				dataset_file = lst[1]
			elif lst[0]=="val_dataset_file":
				val_dataset_file = lst[1]
			elif lst[0]=="array_type":
				array_config['array_type'] = lst[1]
			elif lst[0]=="num_mics":
				array_config['num_mics'] = int(lst[1])
			elif lst[0]=="intermic_dist":
				array_config['intermic_dist'] = float(lst[1])
			elif lst[0]=="room_size":
				array_config['room_size'] = [lst[1], lst[2], lst[3]]
			elif lst[0]=="loss":
				loss_flag = lst[1]
			elif lst[0]=="doa_resolution":
				doa_resolution = int(lst[1])
			else:
				continue
	
	else:
		ref_mic_idx = args.ref_mic_idx
		T60 = None
		SNR = None

		dataset_dtype = args.dataset_dtype
		dataset_condition = args.dataset_condition

		#Loading datasets
		
		dataset_path = args.dataset_file 
		val_dataset_path = args.val_dataset_file 

		noise_simulation = args.noise_simulation

	diffuse_files_path = ""
	
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])
	
	num_doa_classes = math.ceil(181/doa_resolution) if "linear" in array_config['array_type'] else math.ceil(360/doa_resolution)

	SignalAwareDoA_transform = [SignalAwareDoA_features(512, 256, doa_resolution, array_config['array_type'], array_config['array_setup'])]
	train_dataset = SignalAwareDoADataset(dataset_file, transforms=SignalAwareDoA_transform)   #TODO: size = 1000
	dev_dataset = SignalAwareDoADataset(val_dataset_file, transforms=SignalAwareDoA_transform)  # size = 32

	# model
	bidirectional = args.bidirectional
	model = SignalAwareDoA_model(bidirectional, array_config['num_mics'], 257, num_doa_classes, train_dataset, dev_dataset, args.batch_size, args.num_workers)


	## exp path directories
	if dataset_condition=="reverb":
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'
	else:
		loss_flag_str = f'{loss_flag}'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag_str}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'
	exp_name = f'{args.exp_name}' #t60_{T60}_snr_{SNR}dB

	msg_pre_trained = None
	if (not os.path.exists(os.path.join(ckpt_dir,args.resume_model))) and len(args.pre_trained_ckpt_path)>0:
		msg_pre_trained = f"Loading ONLY model parameters from {args.pre_trained_ckpt_path}"
		print(msg_pre_trained)
		ckpt_point = torch.load(args.pre_trained_ckpt_path)
		model.load_state_dict(ckpt_point['state_dict'],strict=False) 

	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir, save_last = True, save_top_k=1, monitor='val_loss')

	model_summary = ModelSummary(max_depth=1)
	early_stopping = EarlyStopping('val_loss', patience=5)
	lr_monitor = LearningRateMonitor(logging_interval='step')

	# training
	precision=16*2
	trainer = pl.Trainer(accelerator='gpu',  num_nodes=args.num_nodes, precision=precision, devices=args.num_gpu_per_node,
					max_epochs = args.max_n_epochs,
					callbacks=[checkpoint_callback, model_summary, lr_monitor, GradNormCallback()],
					logger=tb_logger,
					strategy="ddp_find_unused_parameters_false",
					check_val_every_n_epoch=1,
					log_every_n_steps = 1,
					num_sanity_val_steps=-1,
					profiler="simple",
					fast_dev_run=args.fast_dev_run,
					auto_scale_batch_size=False,
					detect_anomaly=True,
					#gradient_clip_val=5
					)
	
	#trainer.tune(model)
	#print(f'Max batch size fit on memory: {model.batch_size}\n')
				
	msg = f"Train Config: bidirectional: {bidirectional}, T: {T}, doa_resolution: {doa_resolution}, num_doa_classes: {num_doa_classes}, loss_flag: {loss_flag}, precision: {precision}, \n \
		array_type: {array_config['array_type']}, num_mics: {array_config['num_mics']}, intermic_dist: {array_config['intermic_dist']} \n, \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n \
		ref_mic_idx: {ref_mic_idx}, batch_size: {args.batch_size}, ckpt_dir: {ckpt_dir}, exp_name: {exp_name} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)
	if os.path.exists(os.path.join(ckpt_dir,args.resume_model)):
		trainer.fit(model, ckpt_path=os.path.join(ckpt_dir,args.resume_model)) #train_loader, val_loader,
	else:
		trainer.fit(model)#, train_loader, val_loader)

def test(args):

	T = 4
	ref_mic_idx = args.ref_mic_idx
	dataset_file = args.dataset_file
	loss_flag=args.net_type

	# DOA arguments
	doa_tol = args.doa_tol
	doa_euclid_dist = args.doa_euclid_dist
	wgt_mech = args.doa_wgt_mech
	
	se_metrics_flag = False

	if 0:
		T60 = args.T60 
		SNR = args.SNR
		
	else:
		#reading from file for array jobs
		with open(args.input_test_filename, 'r') as f:
			lines = [line for line in f.readlines()]

		T60 = None
		SNR = None
		array_config = {}
		#key value 
		for line in lines:
			lst = line.strip().split()
			if lst[0]=="T60":
				T60 = float(lst[1])
			elif lst[0]=="SNR":
				SNR = float(lst[1])
			elif lst[0]=="array_type":
				array_config['array_type'] = lst[1]
			elif lst[0]=="num_mics":
				array_config['num_mics'] = int(lst[1])
			elif lst[0]=="intermic_dist":
				array_config['intermic_dist'] = float(lst[1])
			elif lst[0]=="room_size":
				array_config['room_size'] = [lst[1], lst[2], lst[3]]
			elif lst[0]=="real_rirs":
				array_config["real_rirs"] = True
			elif lst[0]=="dist":
				array_config["dist"] = int(lst[1])
			elif lst[0]=="doa_resolution":
				doa_resolution = int(lst[1])
			else:
				continue
	
	dataset_condition = args.dataset_condition
	dataset_dtype = args.dataset_dtype
	noise_simulation = args.noise_simulation
	diffuse_files_path = args.diffuse_files_path
	array_config['array_setup'] = get_array_set_up_from_config(array_config['array_type'], array_config['num_mics'], array_config['intermic_dist'])
	num_doa_classes = math.ceil(181/doa_resolution) if "linear" in array_config['array_type'] else math.ceil(360/doa_resolution)


	num_mics = array_config["num_mics"]

	Bohlender_transform = [Bohlender_features(320, 160, doa_resolution, array_config['array_type'], array_config['array_setup'])]


	test_dataset = MovingSourceDataset(dataset_file, array_config,# size=10,
									transforms=Bohlender_transform,
									T60=T60, SNR=SNR, dataset_dtype=dataset_dtype, dataset_condition=dataset_condition,
									noise_simulation=noise_simulation, diffuse_files_path=diffuse_files_path) #
	test_loader = DataLoader(test_dataset, batch_size = args.batch_size,
							 num_workers=args.num_workers, pin_memory=True, drop_last=True)  
	

	if args.dataset_condition =="reverb":
		app_str = f't60_{T60}'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/ref_mic_{ref_mic_idx}'   #{noise_simulation}/
	elif args.dataset_condition =="noisy":
		app_str = f'snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'  #{noise_simulation}
	elif args.dataset_condition =="noisy_reverb":
		app_str = f't60_{T60}_snr_{SNR}dB'
		ckpt_dir = f'{args.ckpt_dir}/{loss_flag}/{dataset_dtype}/{dataset_condition}/{noise_simulation}/ref_mic_{ref_mic_idx}'  #{noise_simulation}
	else:
		app_str = ''

	exp_name = f'Test_{args.exp_name}_{dataset_dtype}_{app_str}'
	
	tb_logger = pl_loggers.TensorBoardLogger(save_dir=ckpt_dir, version=exp_name)
	precision = 32
	trainer = pl.Trainer(accelerator='gpu', precision=precision, devices=args.num_gpu_per_node, num_nodes=args.num_nodes,
						#callbacks=[ Direct_approach_doa_callback(doa_tol) ], 
						logger=tb_logger
						)
	bidirectional = args.bidirectional

	#getting model
	for _file in os.listdir(ckpt_dir):
		if _file.endswith(".ckpt") and _file[:5]=="epoch":
			model_path = _file

	
	msg = f"Test Config: bidirectional: {bidirectional}, T: {T}, batch_size: {args.batch_size}, precision: {precision}, \n \
		ckpt_dir: {ckpt_dir}, exp_name: {exp_name}, \n \
		 doa_resolution: {doa_resolution}, num_doa_classes: {num_doa_classes}, \n \
		model: {model_path}, ref_mic_idx : {ref_mic_idx}, \n \
		dataset_file: {dataset_file}, t60: {T60}, snr: {SNR}, dataset_dtype: {dataset_dtype}, dataset_condition: {dataset_condition}, \n\
		doa_tol: {doa_tol} \n"

	trainer.logger.experiment.add_text("Exp details", msg)

	print(msg)

	if os.path.exists(os.path.join(ckpt_dir, model_path)): 
		model = Bohlender_CNN_LSTM_model.load_from_checkpoint(os.path.join(ckpt_dir, model_path), bidirectional=bidirectional,
					   							num_mics = array_config['num_mics'], num_freq = 161,
												train_dataset=None, val_dataset=None, num_doa_classes=num_doa_classes)

		trainer.test(model, dataloaders=test_loader)
	else:
		print(f"Model path not found in {ckpt_dir}")

if __name__=="__main__":
	#flags
	#torch.autograd.set_detect_anomaly(True)
	args = parser.parse_args()
	if args.train:
		print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
		main(args)
	else:
		print("Testing\n")
		print(f"{torch.cuda.is_available()}, {torch.cuda.device_count()}\n")
		test(args)
