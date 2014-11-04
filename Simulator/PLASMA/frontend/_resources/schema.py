from flask.ext.restful import reqparse,abort
from flask import request
from js_utils import fixKey
from ast import literal_eval
from flask.ext import restful
import wrappers
import schema_helpers
from flask.ext.restful.utils import cors

class DatasetColumnOptions(restful.Resource):
	def get(self):
		return {"datasetColumnOptions": [{"option": option, "id":index} for index,option in enumerate(schema_helpers.types.keys())]}

	def options(self):#http://iamstef.net/ember-cli/using-modules/
		'',200

class EntryCells(restful.Resource):
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "EntryCells GET: "+str(where_args)
		return {'entryCells': [fixKey(entry_cell.to_json()) for entry_cell in wrappers.EntryCell.objects(**where_args)]}

	def options(self):
		'',200

class EntryCell(restful.Resource):
	#IMPLEMENT ME
	pass

class DatasetColumns(restful.Resource):
	def get(self):
		where_args = literal_eval(request.args.get('where','{}'))
		print "DatasetColumns GET: "+str(where_args)
		return {'datasetColumns': [fixKey(dataset_column.to_json()) for dataset_column in wrappers.DatasetColumn.objects(**where_args)]}

	def post(self):
		parser = reqparse.RequestParser()
		parser.add_argument('datasetColumn',type=dict)
		args = parser.parse_args()['datasetColumn']
		print "DatasetColumn POST: "+str(args)

		dataset = wrappers.getDoc('dataset',args['dataset'])

		datasetColumn = wrappers.setDatasetSchemaColumn(dataset,args['order'],args['name'],args['expected'])

		return {'datasetColumn':fixKey(datasetColumn.to_json())},200
	
	def options(self):
		'',200

class DatasetColumn(restful.Resource):
	def get(self,uid):
		return {'datasetColumn': fixKey(wrappers.DatasetColumn.objects(id=uid).get().to_json())}

	def put(self,uid):
		parser = reqparse.RequestParser()
		parser.add_argument('datasetColumn',type=dict)
		args = parser.parse_args()['datasetColumn']
		print "DatasetColumn PUT: "+str(args)

		datasetColumn = wrappers.getDoc('datasetColumn',uid)
		for k,v in args.iteritems():
			if wrappers.getDoc(k,v) is not None:
				datasetColumn[k] = wrappers.getDoc(k,v)
			else:
				datasetColumn[k] = v
		datasetColumn.save()

		return {'datasetColumn':fixKey(datasetColumn.to_json())},200

	def options(self):
		return '',200