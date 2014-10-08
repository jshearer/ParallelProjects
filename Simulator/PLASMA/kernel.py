from database_setup import *
from decl_enum import *

class KernelScope(DeclEnum):
	pre_sim = "presim", "Executed prior to main simulate loop"
	simulate = "sim", "Executed in main simulate loop"
	post_sim = "postsim", "Executed after main simulate loop"

class Kernel(Base):
	__tablename__ = "kernels"
	id = 			Column(Integer, primary_key=True)
	scope = 		Column(KernelScope.db_type())
	type = 			Column(String)
	simulation_id = Column(Integer, ForeignKey("simulations.id"))
	name =	 		Column(String)
	description = 	Column(String)

	__mapper_args__ = {'polymorphic_on':type} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

	simulation = 	relationship("Simulation", backref="kernels")

class Note(Base):
	__tablename__ = "notes"
	id = 			Column(Integer, primary_key=True)
	simulation_id = Column(Integer, ForeignKey("simulations.id"))
	parent_id = 	Column(Integer, ForeignKey(id))
	user_id = 		Column(Integer, ForeignKey("users.id"))
	content = 		Column(String)

	children = 		relationship("Note", backref=backref("parent", remote_side=[id]))
	simulation = 	relationship("Simulation", backref=backref("notes", order_by=id))
	user = 			relationship("User", backref=backref("notes", order_by=id))

class Argument(Base):
	__tablename__ = "arguments"
	id = 			Column(Integer, primary_key=True)
	name =	 		Column(String)
	description = 	Column(String)
	value = 		Column(PickleType)
	parent_id = 	Column(Integer, ForeignKey(id), nullable=True)
	kernel_id = 	Column(Integer, ForeignKey("kernels.id"))

	__mapper_args__ = {'polymorphic_on':name}

	children = 		relationship("Argument", backref=backref("parent", remote_side=[id]))
	kernel = 		relationship("Kernel", backref=backref("arguments", order_by=id))


class Diagnostic(Base):
	__tablename__ = "diagnostics"
	id = 			Column(Integer, primary_key=True)
	name =	 		Column(String)
	content = 		Column(PickleType)
	kernel_id = 	Column(Integer, ForeignKey("kernels.id"))

	__mapper_args__ = {'polymorphic_on':name}

	kernel = 		relationship("Kernel", backref=backref("diagnostics", order_by=id))
