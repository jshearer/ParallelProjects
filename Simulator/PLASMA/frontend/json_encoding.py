from backend.decl_enum import EnumSymbol
from backend.base import Base
from inspect import ismethod
import simplejson as json

def new_alchemy_encoder():
    _visited_objs = []
    class AlchemyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, EnumSymbol):
                return repr(obj)
            if isinstance(obj, Base):
                # an SQLAlchemy class
                fields = {}
                for field in [x for x in dir(obj) if not x.startswith('_') and x != 'metadata']:
                    data = obj.__getattribute__(field)
                    try:
                        json.dumps(data) # this will fail on non-encodable values, like other classes
                        fields[field] = data
                    except TypeError:
                        fields[field] = None
                # a json-encodable dict
                return fields
            if(ismethod(obj)):
                return None
            return obj
    return AlchemyEncoder

alchemy_encoder = new_alchemy_encoder()