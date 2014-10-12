from database_setup import *
from decl_enum import *
import json

class KernelScope(DeclEnum):
	pre_sim = "presim", "Executed prior to main simulate loop"
	simulate = "sim", "Executed in main simulate loop"
	post_sim = "postsim", "Executed after main simulate loop"

class Kernel(Base):
	__tablename__ = "kernels"
	id = 			Column(Integer, primary_key=True)
	scope = 		Column(KernelScope.db_type())
	simulation_id = Column(Integer, ForeignKey("simulations.id"))
	name =	 		Column(String, default="Base")
	description = 	Column(String)

	__mapper_args__ = {"polymorphic_on":name,
					   "polymorphic_identity":"Base"} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

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
	name =	 		Column(String, default="Base")
	description = 	Column(String)
	data = 			Column(PickleType(pickler=json), default=dict())
	parent_id = 	Column(Integer, ForeignKey(id))
	kernel_id = 	Column(Integer, ForeignKey("kernels.id"))

	__mapper_args__ = {"polymorphic_on":name,
					   "polymorphic_identity":"Base"} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

	children = 		relationship("Argument", backref=backref("parent", remote_side=[id]))
	kernel = 		relationship("Kernel", backref=backref("arguments", order_by=id))

	def __init__(self,name="Base",description=None,data=dict(),parent=None,kernel=None):
		self.name = name
		self.description = description
		self.data = data
		self.parent = parent
		self.kernel = kernel

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


class Diagnostic(Base):
	__tablename__ = "diagnostics"
	id = 			Column(Integer, primary_key=True)
	name =	 		Column(String, default="Base")
	content = 		Column(PickleType)
	kernel_id = 	Column(Integer, ForeignKey("kernels.id"))

	__mapper_args__ = {"polymorphic_on":name,
					   "polymorphic_identity":"Base"} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

	kernel = 		relationship("Kernel", backref=backref("diagnostics", order_by=id))
