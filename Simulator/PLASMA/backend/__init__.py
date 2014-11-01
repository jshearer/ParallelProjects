from sqlalchemy import create_engine, event, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import sessionmaker, mapper, relationship, backref
from sqlalchemy.inspection import inspect

engine = create_engine('sqlite:///testing.db', echo=True)
Session = sessionmaker()
Session.configure(bind=engine)
sess = Session()


#this is nessecary so that the defaults defined in the column definitions
#get applied instantly, as opposed to only after database commit.
def instant_defaults_listener(target, args, kwargs):
	for key, column in inspect(target.__class__).columns.items():
		if column.default is not None:
			if callable(column.default.arg):
				setattr(target, key, column.default.arg(target))
			else:
				setattr(target, key, column.default.arg)


event.listen(mapper, 'init', instant_defaults_listener)

Base = declarative_base()

from simulation import *
from kernel import *
from user import *
#Import various polymorphic classes from their folders
# from kernels import *
# from arguments import *
# from diagnostics import *

Base.metadata.bind = engine
Base.metadata.create_all()