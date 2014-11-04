from base import Base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import create_engine, event, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.orm import sessionmaker, scoped_session, mapper, relationship, backref
from decl_enum import *

class User(Base):
	__tablename__ = "users"
	id = 			Column(Integer, primary_key=True)
	name = 			Column(String)
	parent_id = 	Column(Integer, ForeignKey(id))

	children = 		relationship("User", backref=backref("parent", remote_side=[id]))

	'''
	from simulation class:
	user = 		relationship("User", backref=backref("simulations", order_by=id))
	'''

