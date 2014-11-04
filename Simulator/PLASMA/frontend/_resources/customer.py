from flask.ext.restful import reqparse,abort
from flask import request
from js_utils import fixKey
from ast import literal_eval
from flask.ext import restful
import wrappers
from flask.ext.restful.utils import cors

class Customer(restful.Resource):
	
	def get(self,uid):
		return {'customer': fixKey(wrappers.Customer.objects(id=uid).get().to_json())}

	def put(self,uid):
		parser = reqparse.RequestParser()
		parser.add_argument('customer',type=dict)
		args = parser.parse_args().customer
		print "Customer PUT: "+str(args)

		customer = wrappers.getDoc('customer',uid)
		for k,v in args.iteritems():
			if wrappers.getDoc(k,v) is not None:
				customer[k] = wrappers.getDoc(k,v)
			else:
				customer[k] = v
		customer.save()

		return {'customer':fixKey(customer.to_json())},200

	def options(self):
		return '',200

class Customers(restful.Resource):
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "Customer GET: "+str(where_args)
		return {'customers': [fixKey(customer.to_json()) for customer in wrappers.Customer.objects(**where_args)]}

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('customer',type=dict)
		args = parser.parse_args().customer
		print "Customer POST: "+str(args)

		name = args['name']

		users = wrappers.getDoc('user',args.get('users',[]))
		datasets = wrappers.getDoc('dataset', args.get('datasets',[]))

		customer = wrappers.createCustomer(name,datasets=datasets,users=users)

		return {'customer':fixKey(customer.to_json())},200

	def options(self):
		return '',200