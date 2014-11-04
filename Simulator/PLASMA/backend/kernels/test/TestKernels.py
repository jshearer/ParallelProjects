from backend.kernels.Kernel_Bases import *

'''
TODO: Convert to Python 3 so super() is available. 
'''

class PreSimTest(PreSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PreSimulateTest'}

	def __init__(self,**kwargs):
		super(PreSimTest,self).__init__(**kwargs)
		self.name = "PreSimulateTest"

	def execute(self):
		print("Setting kernel test data to 0")
		self.simulation.data['test'] = 0

class SimTest(SimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'SimulateTest'}

	def __init__(self,**kwargs):
		super(SimTest,self).__init__(**kwargs)
		self.name = "SimulateTest"

	def execute(self):
		print("Incrementing test data")
		self.simulation.data['test'] += 1

class PostSimTest(PostSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PostSimulateTest'}

	def __init__(self,**kwargs):
		super(PostSimTest,self).__init__(**kwargs)
		self.name = "PostSimulateTest"

	def execute(self):
		print("Test data is: "+str(self.simulation.data['test']))