from flask.ext.restful import reqparse,abort
from flask import request
from js_utils import fixKey
from ast import literal_eval
from flask.ext import restful
import wrappers
from flask.ext.restful.utils import cors

class Entries(restful.Resource):
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "Entries GET ("+str(type(where_args))+"): "+str(where_args)
		return {'entries': [fixKey(entry.to_json()) for entry in wrappers.Entry.objects(**where_args)]}

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('entry',type=dict)
		args = parser.parse_args().entry
		print "Entry POST"+str(args)

		user = wrappers.getDoc('user',args['user'])
		dataset = wrappers.getDoc('dataset',args['dataset'])
		contents = args['contents'] if type(args['contents']) is dict else [args['contents']]

		try:
			entry = wrappers.createEntry(user,dataset,contents)
			return {'entry': fixKey(entry.to_json())},200
		except ValueError as e:
			return {'err': e.message}, 403

		return {'entry': fixKey(entry.to_json())},200

	def options(self):
		return '',200


class Entry(restful.Resource):
	def get(self,uid):
		return {'entry': fixKey(wrappers.Entry.objects(id=uid).get().to_json())}

	def options(self):
		return '',200