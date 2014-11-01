from PLASMA.simulation import Argument
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, PickleType, ForeignKey

class RangeArgument(Argument):
	__mapper_args__ = {'polymorphic_identity': 'RangeArgument'}

	def __init__(self,arg,range):
		Argument.__init__(self,type="RangeArgument",description="An argument with a defined numeric range.")
		self.arg = arg
		self.range = {"min":min(range),"max":max(range)}

	def validate(self):
		return self.arg <= self.range["max"] and self.arg >= self.range["min"]

	@hybrid_property
	def arg(self):
		return self.data['arg']

	@arg.setter
	def arg(self,arg):
		self.data['arg'] = arg

	@hybrid_property
	def range(self):
	    return self.data['range']

	@range.setter
	def range(self,range):
	    self.data['range'] = range