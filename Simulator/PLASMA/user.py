from database_setup import *
from decl_enum import *

class User(Base):
	__tablename__ = "users"
	id = 			Column(Integer, primary_key=True)
	name = 			Column(String)
	parent_id = 	Column(Integer, ForeignKey(id))

	children = 		relationship("User", backref=backref("parent", remote_side=[id]))

