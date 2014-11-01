from PLASMA.kernels.Kernel_Bases import *
from PLASMA.simulation import *
from PLASMA.arguments.RangeArgument import RangeArgument

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
		self.diagnostic = Diagnostic()
		self.diagnostic.data["test_counter"] = 0
		self.simulation.diagnostics["test_kernel_counting"] = self.diagnostic

		self.range = RangeArgument(range(0,15),0)
		self.simulation.arguments["test_kernel_counting_range"] = self.range

class SimRangeTest(SimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'SimulateRangeTest'}

	def __init__(self):
		SimulateKernel.__init__(self)
		self.name = "SimulateRangeTest"

	def execute(self):
		if(self.range.validate(self.diagnostic.data["test_counter"]+1)):
			self.diagnostic.data["test_counter"]++
			print("Incrementing test data")

class PostSimRangeTest(PostSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PostSimulateRangeTest'}

	def __init__(self):
		PostSimulateKernel.__init__(self)
		self.name = "PostSimulateRangeTest"

	def execute(self):
		print("Test data is: "+str(self.diagnostic.data)+", range is:"+str(self.range))