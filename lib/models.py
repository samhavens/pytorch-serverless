import collections
from .utils import get_labels

import json
import os
import pickle

import numpy as np
from fastai.lm_rnn import get_rnn_classifier

CLASSES = get_labels(os.environ['LABELS_PATH'])
ITOS_NAME = os.environ['ITOS_NAME']

def classification_model(**kwargs):
	with open(ITOS_NAME, 'rb') as pickle_file:
		itos = pickle.load(pickle_file)

	# these magic numbers are from the IMDB notebook (fastai class 2 course 10)
	# same with the function call returned
	bptt,em_sz,nh,nl = 70,400,1150,3
	vs = len(itos)
	dps = np.array([0.4,0.5,0.05,0.3,0.4])*0.5
	c = len(CLASSES)

	return get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
				layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
				dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])
