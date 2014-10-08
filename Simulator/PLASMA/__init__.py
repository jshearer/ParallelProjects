from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy.orm import relationship, backref
from sqlalchemy import Column, Integer, String, PickleType, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_property

engine = create_engine('sqlite:///testing.db', echo=True)
Session = sessionmaker()
Session.configure(bind=engine)
sess = Session()

Base = declarative_base()

from simulation import *
from kernel import *
from user import *
#Import various polymorphic classes from their folders
from kernels import *
from arguments import *
from diagnostics import *

Base.metadata.bind = engine
Base.metadata.create_all()