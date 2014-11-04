from flask.ext.restful import reqparse,abort
from flask import request
from js_utils import fixKey
from ast import literal_eval
from flask.ext import restful
import wrappers
from flask.ext.restful.utils import cors

class User(restful.Resource):
	def get(self,uid):
		return {'user': fixKey(wrappers.User.objects(id=uid).get().to_json())}

	def put(self,uid):
		parser = reqparse.RequestParser()
		parser.add_argument('user',type=dict)
		args = parser.parse_args().user
		print "User PUT: "+str(args)
		print(args)

		user = wrappers.getDoc('user',uid)
		for k,v in args.iteritems():
			if wrappers.getDoc(k,v) is not None:
				user[k] = wrappers.getDoc(k,v)
			else:
				user[k] = v
		user.save()

		return {'user': fixKey(user.to_json())},200

	def options(self):
		return '',200

class Users(restful.Resource):
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "Users GET: "+str(where_args)
		return {'users': [fixKey(user.to_json()) for user in wrappers.User.objects(**where_args)]}

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('user',type=dict)
		args = parser.parse_args().user
		print "User POST: "+str(args)

		newUser = wrappers.createUser(args['username'],args['name'],customers = wrappers.getDoc('customers',args['customers']), datasets = wrappers.getDoc('datasets',args['datasets']))

		return {'user': fixKey(newUser.to_json())},200

	def options(self):
		return '',200
