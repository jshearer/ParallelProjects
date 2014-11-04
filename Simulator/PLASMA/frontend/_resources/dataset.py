from flask.ext.restful import reqparse,abort
from flask import request
from js_utils import fixKey
from ast import literal_eval
from flask.ext import restful
import wrappers
from flask.ext.restful.utils import cors

class Datasets(restful.Resource):
	
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "Dataset GET: "+str(where_args)
		return {'datasets': [fixKey(dataset.to_json()) for dataset in wrappers.Dataset.objects(**where_args)]}

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('dataset',type=dict)
		args = parser.parse_args().dataset
		print "Dataset POST: "+str(args)

		customer = wrappers.getDoc('customer',args['customer'])

		dataset = wrappers.createDataset(customer,args['title'])

		return {'dataset':fixKey(dataset.to_json())},200

	def options(self):
		return '',200

class Dataset(restful.Resource):
	def get(self,uid):
		return {'dataset': fixKey(wrappers.Dataset.objects(id=uid).get().to_json())}

	def put(self,uid):
		parser = reqparse.RequestParser()
		parser.add_argument('dataset',type=dict)
		args = parser.parse_args().dataset
		print "Dataset PUT: "+str(args)

		dataset = wrappers.getDoc('dataset',uid)
		for k,v in args.iteritems():
			if wrappers.getDoc(k,v) is not None:
				dataset[k] = wrappers.getDoc(k,v)
			else:
				dataset[k] = v
		dataset.save()

		return {'dataset':fixKey(dataset.to_json())},200

	def options(self):
		return '',200