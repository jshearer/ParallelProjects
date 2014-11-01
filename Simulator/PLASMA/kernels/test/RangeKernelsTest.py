from PLASMA.kernels.Kernel_Bases import *

'''
TODO: Convert to Python 3 so super() is available. 
'''

class PreSimRangeTest(PreSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PreSimulateRangeTest'}

	def __init__(self):
		PreSimulateKernel.__init__(self)
		self.name = "PreSimulateRangeTest"

	def execute(self):
		print("Setting kernel test data to 0")
		self.simulation.data['test'] = 0

class SimRangeTest(SimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'SimulateRangeTest'}

	def __init__(self):
		SimulateKernel.__init__(self)
		self.name = "SimulateRangeTest"

	def execute(self):
		print("Incrementing test data")
		if(self.simulation.data['test'] in self.simulation.arguments['range']):
			self.simulation.data['test'] += 1

class PostSimRangeTest(PostSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PostSimulateRangeTest'}

	def __init__(self):
		PostSimulateKernel.__init__(self)
		self.name = "PostSimulateRangeTest"

	def execute(self):
		print("Test data is: "+str(self.simulation.data['test'])+", range is:"+str(self.simulation.arguments['range']))