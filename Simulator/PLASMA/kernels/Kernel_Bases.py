from kernel import Kernel, KernelScope
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, PickleType, ForeignKey

class PresimulateKernel(Kernel):
	__mapper_args__ = {'polymorphic_identity': 'PresimulateKernel'}

	def __init__(self):
		Kernel.__init__(self,scope=KernelScope.pre_sim,name="PresimulateKernel")

class SimulateKernel(Kernel):
	__mapper_args__ = {'polymorphic_identity': 'SimulateKernel'}

	def __init__(self):
		Kernel.__init__(self,scope=KernelScope.simulate,name="SimulateKernel")

class PostsimulateKernel(Kernel):
	__mapper_args__ = {'polymorphic_identity': 'PostsimulateKernel'}

	def __init__(self):
		Kernel.__init__(self,scope=KernelScope.post_sim,name="PostsimulateKernel")