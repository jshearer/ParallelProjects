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
	id = 			Column(Integer, primary_key=True)
	step = 			Column(Integer, default=0)
	steps = 		Column(Integer, default=0)
	data = 			Column(PickleType, default=dict)
	state = 		Column(SimulationState.db_type(),default=SimulationState.pre_start)
	parent_id = 	Column(Integer, ForeignKey(id), nullable=True)
	user_id = 		Column(Integer, ForeignKey("users.id"), nullable=True)
	argument_id = 	Column(Integer, ForeignKey("arguments.id"),nullable=False)

	children = 	relationship("Simulation", backref=backref("parent", remote_side=[id]))
	user = 		relationship("User", backref=backref("simulations", order_by=id))
	arguments = relationship("Argument", backref="simulation")

	'''
	from kernel class:
	simulation = 	relationship("Simulation", backref="kernels")
	'''

	def __init__(self,args,steps=0,data=dict,state=SimulationState.pre_start):
		self.arguments = args
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
			presim.execute(self.arguments)
		self.state = SimulationState.initialized

		self.state = SimulationState.running
		for self.step in range(0,self.steps):
			for sim in self.sim_kernels:
				if self.step%sim.after_every == 0:
					sim.execute(self.arguments)

		self.state = SimulationState.finishing
		for postsim in self.post_kernels:
			postsim.execute(self.arguments)

		self.state = SimulationState.finished

class Argument(Base):
	__tablename__ = "arguments"
	id = 			Column(Integer, primary_key=True)
	name =	 		Column(String, default="untitled")
	type = 			Column(String, default="Base")
	description = 	Column(String)
	data = 			Column(PickleType, default=dict)
	parent_id = 	Column(Integer, ForeignKey(id))

	__mapper_args__ = {"polymorphic_on":type,
					   "polymorphic_identity":"Base"} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

	children = 		relationship("Argument", backref=backref("parent", remote_side=[id]))

	'''
	from simulation class
	arguments = relationship("Argument", backref="simulation")
	'''

	'''
	_validate is a recursive function, and exists on all subclasses.
	_validate calls validate, so that validate only has to do actually validation,
	and doesn't have to worry about continuing the recursion.
	'''
	def _validate(self):
		for child in self.children:
			if not child.validate() or (not child._validate()):
				return False
		return True

	def validate(self):
		return self._validate()

	#Return the argument object itself
	def __getitem__(self,key):
		if(self.name==key):
			return self

		for child in self.children:
			try:
				#this is recursive
				child_value = child[key]
				return child_value
			except KeyError:
				#Key not found in this child or its descendants
				pass

		#Key not found
		raise KeyError("No matching argument found in tree")

	def __setitem__(self,key,value):
		value.name = key
		try:
			preexisting = self[key]
			raise KeyError("An argument with that name already exists")
		except KeyError:
			#No arguments with that name exist already
			self.children.append(value)

