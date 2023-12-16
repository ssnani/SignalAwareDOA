
import numpy as np
import random

class Parameter:
	""" Random parammeter class.
	You can indicate a constant value or a random range in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and vectors.
	"""
	def __init__(self, *args):
		if len(args) == 1:
			self.random = False
			self.value = np.array(args[0])
			self.min_value = None
			self.max_value = None
		elif len(args) == 2:
			self. random = True
			self.min_value = np.array(args[0])
			self.max_value = np.array(args[1])
			self.value = None
		else: 
			raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
	
	def getValue(self, round_format: int):
		if self.random:
			val =  self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
			self.value = np.round(val, round_format)
			return self.value
		else:
			return self.value

class ParameterSet:
	""" Random parammeter class.
	You can indicate a constant value or a random set in its constructor and then
	get a value acording to that with getValue(). It works with both scalars and list.
	"""
	def __init__(self, lst):

		num_choices = len(lst)
		self.lst = lst
	
	def getValue(self): 
		return random.choice(self.lst)
