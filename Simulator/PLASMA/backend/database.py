from sqlalchemy import create_engine, event, Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, backref

engine = create_engine('sqlite:///testing.db', echo=True)

Session = sessionmaker(autocommit=False,
                       autoflush=False,
                       bind=engine)

db_session = scoped_session(Session)

def init_db():
	#initialize all of the models and identities
	from simulation import Simulation, Argument, Diagnostic
	from kernel import Kernel, Note
	from user import User
	from kernels import *
	from arguments import *
	from diagnostics import *
	from base import Base
	Base.metadata.create_all(engine)

init_db()