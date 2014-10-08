from database_setup import *
from decl_enum import *

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
	step = 		Column(Integer)
	steps = 	Column(Integer)
	state = 	Column(SimulationState.db_type())
	parent_id = Column(Integer, ForeignKey(id), nullable=True)
	user_id = 	Column(Integer, ForeignKey("users.id"))

	children = 	relationship("Simulation", backref=backref("parent", remote_side=[id]))
	user = 		relationship("User", backref=backref("simulations", order_by=id))

	@hybrid_property
	def pre_kernels(self):
	    return tuple(kernel for kernel in self.kernels if kernel.scope is KernelScope.pre_sim)
	@pre_kernels.expression
	def pre_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.pre_sim)

	@hybrid_property
	def sim_kernels(self):
	    return tuple(kernel for kernel in self.kernels if kernel.scope is KernelScope.simulate)
	@sim_kernels.expression
	def sim_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.simulate)

	@hybrid_property
	def post_kernels(self):
	    return tuple(kernel for kernel in self.kernels if kernel.scope is KernelScope.post_sim)
	@post_kernels.expression
	def post_kernels():
	    return select([Kernel]).where(Kernel.scope==KernelScope.post_sim)