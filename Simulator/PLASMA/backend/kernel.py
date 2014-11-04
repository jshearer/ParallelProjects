from base import Base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import create_engine, event, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.orm import sessionmaker, scoped_session, mapper, relationship, backref
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
	after_every = 	Column(Integer, default=1)
	description = 	Column(String)

	__mapper_args__ = {"polymorphic_on":name,
					   "polymorphic_identity":"Base"} #check http://techspot.zzzeek.org/2011/01/14/the-enum-recipe/ at the end to see why this is awesome

	simulation = 	relationship("Simulation", backref="kernels")

	def execute():
		raise NotImplementedError("The kernel base class cannot be executed.")

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
