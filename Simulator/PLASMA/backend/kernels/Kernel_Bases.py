from PLASMA.kernel import Kernel, KernelScope
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, PickleType, ForeignKey

class PreSimulateKernel(Kernel):
	def __init__(self):
		self.scope = KernelScope.pre_sim

class SimulateKernel(Kernel):
	def __init__(self):
		self.scope = KernelScope.simulate

class PostSimulateKernel(Kernel):
	def __init__(self):
		self.scope = KernelScope.post_sim