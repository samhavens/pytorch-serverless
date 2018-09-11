try:
	import unzip_requirements
except ImportError:
	pass

import collections, os, pickle, json, traceback
import urllib.parse
import numpy as np

from lib.models import classification_model
from lib.utils import download_file, get_labels
from fastai.core import VV, to_np
from torch import topk # sometimes linter compains DNE but it does
from torch import load as toad


BUCKET_NAME = os.environ['BUCKET_NAME']
STATE_DICT_NAME = os.environ['STATE_DICT_NAME']
ITOS_NAME = os.environ['ITOS_NAME']


def load_model(m, p):
	'''
	copy-pasted from fast.ai core
	'''
    sd = toad(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)

class SetupModel(object):
	model = classification_model()
	labels = get_labels(os.environ['LABELS_PATH'])
	# might need stoi shit here

	def __init__(self, f):
		self.f = f
		file_path = f'/tmp/{STATE_DICT_NAME}'
		download_file(BUCKET_NAME, STATE_DICT_NAME, file_path)
		load_model(self.model, file_path)
		os.remove(file_path)
		#set batch size to 1
		self.model[0].bs=1
		#turn off dropout
		self.model.eval()
		#reset hidden state
		self.model.reset()

		self.stoi = self.setup_stoi()

	def setup_stoi(self):
		file_path = f'/tmp/{ITOS_NAME}'
		download_file(BUCKET_NAME, ITOS_NAME, file_path)
		_itos = pickle.load(file_path)
		stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(_itos)})
		os.remove(file_path)
		return stoi

	def __call__(self, *args, **kwargs):
		return self.f(*args, **kwargs)


def build_pred(label_idx, log, prob):
	label = SetupModel.labels[label_idx]
	return dict(label=label, log=float(log), prob=float(prob))


def parse_params(params):
	text = urllib.parse.unquote_plus(params.get('text', ''))
	n_labels = len(SetupModel.labels)
	top_k = int(params.get('top_k', 3))
	if top_k < 1: top_k = n_labels
	return dict(text=text, top_k=min(top_k, n_labels))


def predict(text):
	idxs = np.array([[SetupModel.stoi[p] for p in text.strip().split(" ")]])
	idxs = np.transpose(idxs)
	inp = m(VV(idxs))
	return SetupModel.model(inp).mean(0)


@SetupModel
def handler(event, _):
	if event is None: event = {}
	print(event)
	try:
		# keep the lambda function warm
		if event.get('detail-type') is 'Scheduled Event':
			return 'nice & warm'

		params = parse_params(event.get('queryStringParameters', {}))
		out = predict(params['text'])
		top = out.topk(params.get('top_k'), sorted=True)

		logs, idxs = (t.data.numpy() for t in top)
		probs = np.exp(logs)
		preds = [build_pred(idx, logs[i], probs[i]) for i, idx in enumerate(idxs)]

		response_body = dict(predictions=preds)
		response = dict(statusCode=200, body=response_body)

	except Exception as e:
		response_body = dict(error=str(e), traceback=traceback.format_exc())
		response = dict(statusCode=500, body=response_body)

	response['body'] = json.dumps(response['body'])
	print(response)
	return response
