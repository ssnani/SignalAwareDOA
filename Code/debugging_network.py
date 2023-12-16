import torch
import torch.nn as nn

def isNanisInf(x):
	is_NaN, is_Inf =  torch.isnan(x).any(), torch.isinf(x).any()

	#print(f'isNaN: {is_NaN}, isInf: {is_Inf}' )

	if is_NaN or is_Inf:
		print(f'isNaN: {is_NaN}, isInf: {is_Inf}' )
		print(torch.where(torch.isinf(x)))
		print(torch.where(torch.isnan(x)))
		breakpoint()
		#torch.save(x, '../Logs/grad_instable.pt')
		pass

"""
	******* hook functionality ****
	Debugging: Observe intermediate gradients *** during *** backward call

	Tensors:
		- backward : register_hook( hook_fun_tensor )
			- hook function has input as tensor's grad

		- Like Manual debugging, 
			- add next to the tensor variable of interest x, x.register_hook(lambda grad : print(grad))
	
	Layers:
		- forward : for layers inside a module register_forward_hook( hook_fn_lyr )
"""

def hook_fun(layer, input, output):
	#isNanisInf = lambda x: print(f' layer: {layer}, is NaN: {torch.isnan(x).any()} , isInf: {torch.isinf(x).any()}' )

	print(f"layer: {layer}")
	print(f"Input: {input[0].dtype}, {input[0].shape}, is_NaN: {torch.isnan(input[0]).any()}, isInf: {torch.isinf(input[0]).any()}")
	#isNanisInf(input[0])
	if isinstance(output, tuple):
		output = output[0]

	print(f"Output: {output.dtype}, {output.shape}, is_NaN: {torch.isnan(output).any()}, isInf: {torch.isinf(output).any()}")
	#isNanisInf(output)
	
	if output.requires_grad:
		output.register_hook( isNanisInf ) 
	else:
		print(f"{layer} output Grad Not Required")



def recursive_layer_forward_hook(module):
	#print(module)
	if isinstance(module, nn.Sequential):
		for layer in module:
			layer.register_forward_hook( hook_fun)



def get_all_layers(module):
	for name, layer in module._modules.items():
	#If it is a sequential, don't register a hook on it
	# but recursively register hook on all it's module children
		if isinstance(layer, nn.Sequential):
			get_all_layers(layer)
		else:
	  		# it's a non sequential. Register a hook
	  		layer.register_forward_hook( hook_fun )