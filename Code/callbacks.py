import torch 
import torchaudio
import torchaudio.transforms as T
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT

class GradNormCallback(Callback):
	"""
	Logs the gradient norm.
	"""

	def ___on_after_backward(self, trainer: "pl.Trainer", model): 
		grad_norm_dict = gradient_norm_per_layer(model)
		self.log_dict(grad_norm_dict)

	def on_before_optimizer_step(self, trainer, model, optimizer, optimizer_idx=0): #trainer, model, 
		grad_norm_dict, grad_data_ratio = gradient_norm_per_layer(model)
		#print(grad_norm_dict, grad_data_ratio)
		self.log_dict(grad_norm_dict)
		#self.log_dict(grad_data_ratio)

def gradient_norm_per_layer(model):
	total_norm = {}
	grad_data_ratio = {}
	for layer_name, param in model.named_parameters():
		#breakpoint()
		if param.grad is not None:
			#print(param.grad)
			param_grad_norm = param.grad.detach().data.norm(2)
			param_norm = param.data.norm(2)
			total_norm[layer_name] = param_grad_norm #.item() ** 2

			param_std = torch.std(param.data)
			param_grad_std = torch.std(param.grad.detach().data)
			grad_data_ratio[layer_name] = param_grad_std/param_std
	#breakpoint()
	#total_norm = total_norm ** (1. / 2)
	return total_norm, grad_data_ratio