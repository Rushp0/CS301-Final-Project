# milestone 4 model compression code

""" Knowledge Distillation is not implemented for Tensorflow, using automatic 
    https://nni.readthedocs.io/en/v1.6/TrialExample/KDExample.html
"""

# imports
from keras.models import load_model
from  tensorflow.keras import metrics
import tensorflow as tf
from nni.compression.tensorflow import Pruner


model = load_model("milestone-3-v2", compile=False, custom_objects=None)
config_list = [{'sparsity': 0.8, 'op_types': ['default'] }]
pruner = Pruner(model, config_list)

# following quickstart guide on https://nni.readthedocs.io/en/v1.9/Compression/QuickStart.html
pruner.compress() # throws NotImplemented Error despite following above quickstart guide. NotImplementedError: Pruners must overload calc_masks()

pruner.export_model('compressed_model')