from backend.kernel import Kernel, KernelScope

class PreSimulateKernel(Kernel):
	def __init__(self,**kwargs):
		
		super(PreSimulateKernel,self).__init__(**kwargs)
		self.scope = KernelScope.pre_sim

class SimulateKernel(Kernel):
	def __init__(self,**kwargs):
		
		super(SimulateKernel,self).__init__(**kwargs)
		self.scope = KernelScope.simulate

class PostSimulateKernel(Kernel):
	def __init__(self,**kwargs):
		
		super(PostSimulateKernel,self).__init__(**kwargs)
		self.scope = KernelScope.post_sim