from database_setup import *
from decl_enum import *
from kernel import KernelScope

class SimulationState(DeclEnum):
	container = "container", "A simulation that will never be run. Used for grouping simulations."
	pre_start = "pre-start", "Instantiated, but pre-sim methods not run"
	initializing = "initializing", "Pre-sim methods running"
	initialized = "initialized", "Pre-sim methods done"
	running = "running", "Simulate loop in progress"
	paused = "paused", "Simulation loop paused but still alive"
	stopped = "stopped", "Simulation was canceled"
	error = "error", "Simulation encountered an error"
	finishing = "finishing", "Simulate loop done, running post-sim methods"
	finished = "finished", "Post-sim methods complete"

class Simulation(Base):
	__tablename__ = "simulations"
	id = 		Column(Integer, primary_key=True)
	step = 		Column(Integer, default=0)
	steps = 	Column(Integer, default=0)
	data = 		Column(PickleType, default=dict())
	state = 	Column(SimulationState.db_type(),default=SimulationState.pre_start)
	parent_id = Column(Integer, ForeignKey(id), nullable=True)
	user_id = 	Column(Integer, ForeignKey("users.id"))

	children = 	relationship("Simulation", backref=backref("parent", remote_side=[id]))
	user = 		relationship("User", backref=backref("simulations", order_by=id))
	'''
	from kernel class:
	simulation = 	relationship("Simulation", backref="kernels")
	'''

	def __init__(self,steps=0,data=dict(),state=SimulationState.pre_start):
		self.steps = steps
		self.data = data
		self.state = state

	@hybrid_property
	def pre_kernels(self):
	    return [kernel for kernel in self.kernels if kernel.scope is KernelScope.pre_sim]
	@pre_kernels.expression
	def pre_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.pre_sim)

	@hybrid_property
	def sim_kernels(self):
	    return [kernel for kernel in self.kernels if kernel.scope is KernelScope.simulate]
	@sim_kernels.expression
	def sim_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.simulate)

	@hybrid_property
	def post_kernels(self):
	    return [kernel for kernel in self.kernels if kernel.scope is KernelScope.post_sim]
	@post_kernels.expression
	def post_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.post_sim)

	def run(self):
		self.state = SimulationState.initializing
		for presim in self.pre_kernels:
			presim.execute()
		self.state = SimulationState.initialized

		self.state = SimulationState.running
		for self.step in range(0,self.steps):
			for sim in self.sim_kernels:
				if self.step%sim.after_every == 0:
					sim.execute()

		self.state = SimulationState.finishing
		for postsim in self.post_kernels:
			postsim.execute()

		self.state = SimulationState.finished


