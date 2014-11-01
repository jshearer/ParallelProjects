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

	def execute(self,args,diagnostics):
		print("Setting kernel test data to 0")
		diagnostic = Diagnostic()
		diagnostic.data = 0
		diagnostics["counter"] = diagnostic

		range_arg = RangeArgument(range(0,15),0)
		args["range"] = range_arg

class SimRangeTest(SimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'SimulateRangeTest'}

	def __init__(self):
		SimulateKernel.__init__(self)
		self.name = "SimulateRangeTest"

	def execute(self,args,diagnostics):
		if(args["range"].validate(diagnostics["counter"].data+1)):
			diagnostics["counter"].data = diagnostics["counter"].data + 1
			print("Incrementing test data")

class PostSimRangeTest(PostSimulateKernel):
	__mapper_args__ = {'polymorphic_identity': 'PostSimulateRangeTest'}

	def __init__(self):
		PostSimulateKernel.__init__(self)
		self.name = "PostSimulateRangeTest"

	def execute(self,args,diagnostics):
		print("Test data is: "+str(diagnostics["counter"].data)+", range is:"+str(args["range"]))