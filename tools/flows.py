import math
import types

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# first, we get the estimation of distribution by Masked Autoencoder 
# for Distribution Estimation
# For a normal neural newwork, the neigbhour later's neurons will all 
# be connected, but in MADE, some connection are given up and thus the neuron 

# this MADE constrains the normal auto encoder so that we can learn the distribution
# by using joint probabillity with conditional dependence
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
	"""
	mask_type: input | None | output
	
	See Figure 1 for a better illustration:
	https://arxiv.org/pdf/1502.03509.pdf
	"""
	if mask_type == 'input':
		in_degrees = torch.arange(in_features) % in_flow_features
	else:
		in_degrees = torch.arange(in_features) % (in_flow_features - 1)

	if mask_type == 'output':
		out_degrees = torch.arange(out_features) % in_flow_features - 1
	else:
		out_degrees = torch.arange(out_features) % (in_flow_features - 1)

	return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
	def __init__(self,
				 in_features,
				 out_features,
				 mask,
				 cond_in_features=None,
				 bias=True):
		super(MaskedLinear, self).__init__()
		self.linear = nn.Linear(in_features, out_features)
		if cond_in_features is not None:
			self.cond_linear = nn.Linear(
				cond_in_features, out_features, bias=False)

		self.register_buffer('mask', mask)

	def forward(self, inputs, cond_inputs=None):
		output = F.linear(inputs, self.linear.weight * self.mask,
						  self.linear.bias)
		if cond_inputs is not None:
			output += self.cond_linear(cond_inputs)
		return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
	""" An implementation of MADE
	(https://arxiv.org/abs/1502.03509s).
	"""

	def __init__(self,
				 num_inputs,
				 num_hidden,
				 num_cond_inputs=None,
				 s_act='tanh',
				 t_act='relu',
				 pre_exp_tanh=False):
		super(MADESplit, self).__init__()

		self.pre_exp_tanh = pre_exp_tanh

		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

		input_mask = get_mask(num_inputs, num_hidden, num_inputs,
							  mask_type='input')
		hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
		output_mask = get_mask(num_hidden, num_inputs, num_inputs,
							   mask_type='output')

		act_func = activations[s_act]
		self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
									  num_cond_inputs)

		self.s_trunk = nn.Sequential(act_func(),
								   nn.MaskedLinear(num_hidden, num_hidden,
												   hidden_mask), act_func(),
								   nn.MaskedLinear(num_hidden, num_inputs,
												   output_mask))

		act_func = activations[t_act]
		self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
									  num_cond_inputs)

		self.t_trunk = nn.Sequential(act_func(),
								   nn.MaskedLinear(num_hidden, num_hidden,
												   hidden_mask), act_func(),
								   nn.MaskedLinear(num_hidden, num_inputs,
												   output_mask))
		
	def forward(self, inputs, cond_inputs=None, mode='direct'):
		if mode == 'direct':
			h = self.s_joiner(inputs, cond_inputs)
			m = self.s_trunk(h)
			
			h = self.t_joiner(inputs, cond_inputs)
			a = self.t_trunk(h)

			if self.pre_exp_tanh:
				a = torch.tanh(a)
			
			u = (inputs - m) * torch.exp(-a)
			return u, -a.sum(-1, keepdim=True)

		else:
			x = torch.zeros_like(inputs)
			for i_col in range(inputs.shape[1]):
				h = self.s_joiner(x, cond_inputs)
				m = self.s_trunk(h)

				h = self.t_joiner(x, cond_inputs)
				a = self.t_trunk(h)

				if self.pre_exp_tanh:
					a = torch.tanh(a)

				x[:, i_col] = inputs[:, i_col] * torch.exp(
					a[:, i_col]) + m[:, i_col]
			return x, -a.sum(-1, keepdim=True)

class MADE(nn.Module):
	""" An implementation of MADE
	(https://arxiv.org/abs/1502.03509s).
	"""

	def __init__(self,
				 num_inputs,
				 num_hidden,
				 num_cond_inputs=None,
				 act='relu',
				 pre_exp_tanh=False):
		super(MADE, self).__init__()

		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
		act_func = activations[act]

		input_mask = get_mask(
			num_inputs, num_hidden, num_inputs, mask_type='input')
		hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
		output_mask = get_mask(
			num_hidden, num_inputs * 2, num_inputs, mask_type='output')

		self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
									  num_cond_inputs)

		self.trunk = nn.Sequential(act_func(),
								   nn.MaskedLinear(num_hidden, num_hidden,
												   hidden_mask), act_func(),
								   nn.MaskedLinear(num_hidden, num_inputs * 2,
												   output_mask))

	def forward(self, inputs, cond_inputs=None, mode='direct'):
		if mode == 'direct':
			h = self.joiner(inputs, cond_inputs)
			m, a = self.trunk(h).chunk(2, 1)
			if torch.isnan(a).any():
				print("NaNs in `a` before exp (mode='direct')")
				raise ValueError("NaNs in `a` before exp")
			u = (inputs - m) * torch.exp(-a)
			return u, -a.sum(-1, keepdim=True)

		else:
			x = torch.zeros_like(inputs)
			for i_col in range(inputs.shape[1]):
				h = self.joiner(x, cond_inputs)
				m, a = self.trunk(h).chunk(2, 1)
				if torch.isnan(a).any():
					print(f"NaNs in `a` before exp (mode='inverse'), i_col={i_col}")
					raise ValueError("NaNs in `a` before exp")
				x[:, i_col] = inputs[:, i_col] * torch.exp(
					a[:, i_col]) + m[:, i_col]
			return x, -a.sum(-1, keepdim=True)



class BatchNormFlow(nn.Module):
	""" An implementation of a batch normalization layer from
	Density estimation using Real NVP
	(https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
		super(BatchNormFlow, self).__init__()

		self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
		self.beta = nn.Parameter(torch.zeros(num_inputs))
		self.momentum = momentum
		self.eps = eps

		self.register_buffer('running_mean', torch.zeros(num_inputs))
		self.register_buffer('running_var', torch.ones(num_inputs))

	def forward(self, inputs, cond_inputs=None, mode='direct'):
		if mode == 'direct':
			if self.training:
				self.batch_mean = inputs.mean(0)
				self.batch_var = (
					inputs - self.batch_mean).pow(2).mean(0) + self.eps

				self.running_mean.mul_(self.momentum)
				self.running_var.mul_(self.momentum)

				self.running_mean.add_(self.batch_mean.data *
									   (1 - self.momentum))
				self.running_var.add_(self.batch_var.data *
									  (1 - self.momentum))

				mean = self.batch_mean
				var = self.batch_var
			else:
				mean = self.running_mean
				var = self.running_var

			x_hat = (inputs - mean) / var.sqrt()
			y = torch.exp(self.log_gamma) * x_hat + self.beta
			return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
				-1, keepdim=True)
		else:
			if self.training:
				mean = self.batch_mean
				var = self.batch_var
			else:
				mean = self.running_mean
				var = self.running_var

			x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

			y = x_hat * var.sqrt() + mean

			return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
				-1, keepdim=True)


class ActNorm(nn.Module):
	""" An implementation of a activation normalization layer
	from Glow: Generative Flow with Invertible 1x1 Convolutions
	(https://arxiv.org/abs/1807.03039).
	"""

	def __init__(self, num_inputs):
		super(ActNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(num_inputs))
		self.bias = nn.Parameter(torch.zeros(num_inputs))
		self.initialized = False

	def forward(self, inputs, cond_inputs=None, mode='direct'):
		if self.initialized == False:
			self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
			self.bias.data.copy_(inputs.mean(0))
			self.initialized = True

		if mode == 'direct':
			return (
				inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
					-1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
		else:
			return inputs * torch.exp(
				-self.weight) + self.bias, -self.weight.sum(
					-1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class Reverse(nn.Module):
	""" An implementation of a reversing layer from
	Density estimation using Real NVP
	(https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs):
		super(Reverse, self).__init__()
		self.perm = np.array(np.arange(0, num_inputs)[::-1])
		self.inv_perm = np.argsort(self.perm)

	def forward(self, inputs, cond_inputs=None, mode='direct'):
		if mode == 'direct':
			return inputs[:, self.perm], torch.zeros(
				inputs.size(0), 1, device=inputs.device)
		else:
			return inputs[:, self.inv_perm], torch.zeros(
				inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self,
				 num_inputs,
				 num_hidden,
				 mask,
				 num_cond_inputs=None,
				 s_act='tanh',
				 t_act='relu'):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask

		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
		s_act_func = activations[s_act]
		t_act_func = activations[t_act]

		if num_cond_inputs is not None:
			total_inputs = num_inputs + num_cond_inputs
		else:
			total_inputs = num_inputs
			
		self.scale_net = nn.Sequential(
			nn.Linear(total_inputs, num_hidden), s_act_func(),
			nn.Linear(num_hidden, num_hidden), s_act_func(),
			nn.Linear(num_hidden, num_inputs))
		self.translate_net = nn.Sequential(
			nn.Linear(total_inputs, num_hidden), t_act_func(),
			nn.Linear(num_hidden, num_hidden), t_act_func(),
			nn.Linear(num_hidden, num_inputs))

		def init(m):
			if isinstance(m, nn.Linear):
				m.bias.data.fill_(0)
				nn.init.orthogonal_(m.weight.data)

	def forward(self, inputs, cond_inputs=None, mode='direct'):
		mask = self.mask
		
		masked_inputs = inputs * mask
		if cond_inputs is not None:
			masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)
		
		if mode == 'direct':
			log_s = self.scale_net(masked_inputs) * (1 - mask)
			t = self.translate_net(masked_inputs) * (1 - mask)
			s = torch.exp(log_s)
			return inputs * s + t, log_s.sum(-1, keepdim=True)
		else:
			log_s = self.scale_net(masked_inputs) * (1 - mask)
			t = self.translate_net(masked_inputs) * (1 - mask)
			s = torch.exp(-log_s)
			return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class FlowSequential(nn.Sequential):
	""" A sequential container for flows.
	In addition to a forward pass it implements a backward pass and
	computes log jacobians.
	"""

	def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
		self.num_inputs = inputs.size(-1)

		if logdets is None:
			logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

		assert mode in ['direct', 'inverse']

		modules = self._modules.values() if mode == 'direct' else reversed(self._modules.values())

		for i, module in enumerate(modules):
			if torch.isnan(inputs).any():
				print(f"NaN detected in inputs before module {i} ({module.__class__.__name__})")
				raise ValueError("NaN before module")

			inputs, logdet = module(inputs, cond_inputs, mode)

			if torch.isnan(inputs).any():
				print(f"NaN in outputs after module {i} ({module.__class__.__name__})")
				raise ValueError("NaN in inputs after module")

			if torch.isnan(logdet).any():
				print(f"NaN in logdet after module {i} ({module.__class__.__name__})")
				raise ValueError("NaN in logdet")

			logdets += logdet

		return inputs, logdets


	def log_probs(self, inputs, cond_inputs = None):

		# see if nan in inputs or cond_inputs
		if torch.isnan(inputs).any():
			print(inputs[torch.isnan(inputs)])
			print(torch.isnan(inputs).sum())
			raise ValueError('Nan in inputs')
		if cond_inputs is not None and torch.isnan(cond_inputs).any():
			print(cond_inputs[torch.isnan(cond_inputs)])
			print(torch.isnan(cond_inputs).sum())
			raise ValueError('Nan in cond_inputs')
			
		u, log_jacob = self(inputs, cond_inputs)
		
		if torch.isnan(u).any():
			print(inputs[torch.isnan(u)])
			print(torch.isnan(u).sum())
			raise ValueError('Nan in u')
		if torch.isnan(log_jacob).any():
			raise ValueError('Nan in log_jacob')
			
		
		log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
			-1, keepdim=True)
		return (log_probs + log_jacob).sum(-1, keepdim=True)

	def sample(self, num_samples=None, noise=None, cond_inputs=None):
		if noise is None:
			noise = torch.Tensor(num_samples, self.num_inputs).normal_()
		device = next(self.parameters()).device
		noise = noise.to(device)
		if cond_inputs is not None:
			cond_inputs = cond_inputs.to(device)
		samples = self.forward(noise, cond_inputs, mode='inverse')[0]
		return samples
    
    
# Masked Autoregressive flow for density / distribution estimate
# 
class MAF(nn.Module):
	def __init__(self, num_inputs, num_cond_inputs, num_hidden, num_blocks, act):
		super(MAF, self).__init__()
		modules = []
		for _ in range(num_blocks):
			modules += [
				MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
				BatchNormFlow(num_inputs),
				Reverse(num_inputs)
				]
			self.Flow = FlowSequential(*modules)
			
			for module in self.Flow.modules():
				if isinstance(module, nn.Linear):
					nn.init.orthogonal_(module.weight)
					if hasattr(module, 'bias') and module.bias is not None:
						module.bias.data.fill_(0)

	def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
		inputs,logdets = self.Flow(inputs,cond_inputs,mode,logdets)
		return inputs, logdets

	def log_probs(self, inputs, cond_inputs = None):
		output = self.Flow.log_probs(inputs,cond_inputs)
		return output

	def sample(self, num_samples=None, noise=None, cond_inputs=None):
		samples = self.Flow.sample(num_samples,noise,cond_inputs)
		return samples


