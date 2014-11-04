from sqlalchemy import event
from sqlalchemy.orm import mapper
from sqlalchemy.inspection import inspect
from sqlalchemy.exc import InvalidRequestError

from database import db_session

#this is nessecary so that the defaults defined in the column definitions
#get applied instantly, as opposed to only after database commit.
@event.listens_for(mapper,'init')
def instant_defaults_listener(target, args, kwargs):
	for key, column in inspect(target.__class__).columns.items():
		if column.default is not None:
			if callable(column.default.arg):
				setattr(target, key, column.default.arg(target))
			else:
				setattr(target, key, column.default.arg)
	try:
		db_session.add(target)
	except InvalidRequestError:
		pass