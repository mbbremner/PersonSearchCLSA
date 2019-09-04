# ============================================================
# ------------< Pre-trained Model Initialization   >----------
# ============================================================

# This Script initializes pre-trained models in the proper format

# ----------------- Imports ------------------

import mxnet as mx
from mxnet.gluon import nn
from mxnet import module as mod
import mxnet.ndarray as nd
import re as regex
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import io as IO
import numpy as np
import argparse
import time
from scipy import misc


# ------------------------------------------------------------
# ---------------------< Functions >--------------------------
# ------------------------------------------------------------

# Application argument parser
def create_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('-N', nargs=1, type=int, default=[13003])
    ap.add_argument('-dense_size', type=int, nargs=1, default=[256])
    ap.add_argument('-batch_size', type=int, nargs=1, default=[32])
    ap.add_argument('-outfile', type=str, nargs=1, default=['Models/Res50_Baseline_Modified'])
    ap.add_argument('-param_in', type=str, nargs=1, default=['Models/Res50_pretrained_model-0000.params'])
    ap.add_argument('-sym_in', type=str, nargs=1, default=['Models/Res50_pretrained_model-symbol.json'])
    return ap


# Input Transform
def transform(data):

    # data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    # data.astype('float32')
    return (data / 255 - rgb_mean) / rgb_std


# Get data iterators REAL function
def get_iters(batch_size):
    train_iter = mx.image.ImageIter(
        batch_size=batch_size,
        data_shape=(3, 128, 256),
        label_width=1,
        path_imgrec='Data/train.rec',
        path_imglist='Data/train.lst'
    )

    test_iter = mx.image.ImageIter(
        batch_size=batch_size,
        data_shape=(3, 128, 256),
        label_width=1,
        path_imgrec='Data/test.rec',
        path_imglist='Data/test.lst'
    )

    val_iter = mx.image.ImageIter(
        batch_size=batch_size,
        data_shape=(3, 128, 256),
        label_width=1,
        path_imgrec='Data/val.rec',
        path_imglist='Data/val.lst'
    )

    return train_iter, test_iter, val_iter


# Top Welcome Banner
def print_welcome():
    print(" =========================================================================")
    print("                     Constructing Baseline Model                          ")
    print(" =========================================================================\n")

# Takes argparser opts as input, displays applic
# ation arguments in terminal
def print_application_arguments(options):
    print("\n Application Arguments: \n")
    for opt in vars(options):
        print("   - " + str(opt) + " " + str(getattr(options, opt)))
    print('\n\n')


# ------------------------------------------------------------
# ---------------------------< Script >-----------------------
# ------------------------------------------------------------

argparser = create_argparser()                  # Initialize argument parser
opts = argparser.parse_args()                   # Parse input arguments
print_welcome()                                 # Welcome banner upon execution
print_application_arguments(opts)               # Display input arguments
split_pattern = "[ <>]"                         # splits model internals for console output


param_file = opts.param_in[0]                   # Pretrained params
sym_file = opts.sym_in[0]                       # Pretrained Symbol File
out_path = opts.outfile[0]                      # Model out path
FC_len = opts.dense_size[0]                      # Size Of Dense Layer
N = opts.N[0]
batch_size = opts.batch_size[0]

cpu_ctx = mx.cpu()                                      # CPU context
gpu_ctx = mx.gpu()                                      # GPU context
train_iter, val_iter, _ = get_iters(batch_size)         # Fetch Iterators


# Obtain feed-forward data
for b, batch in enumerate(train_iter):
    data = batch.data[0].as_in_context(cpu_ctx)
    break

# Transfer features from here
resnet50 = vision.resnet50_v1(pretrained=True, ctx=cpu_ctx)             # Load pre-trained model
resnet50.hybridize()

# Transfer pre-trained features to a new net
net = vision.resnet50_v1(classes=N, prefix='resnetv10_')
with net.name_scope():
    net.output = nn.Dense(FC_len, flatten=True)
    net.output.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
net.features = resnet50.features
net.hybridize()
net(data)
net.export('UntrainedModels/Res50_pretrained_model')

# Load Resnet 50 Model
sym50 = mx.sym.load(sym_file)                                           # Load symbol file
feature_layer = 'FC1_output'                                            # Desired output layer
sym_layer = mx.symbol.FullyConnected(name='FC1', num_hidden=N)          # Define dense layer
composed = sym_layer(FC1_data=sym50, name='FC1')                        # Combine with loaded symbol model
internals = composed.get_internals()                                    # Fetch internals, for verification


# Bring together network and feed forward
net = gluon.nn.SymbolBlock(outputs=internals[feature_layer], inputs=mx.sym.var('data'))
net.collect_params('^FC').initialize(mx.init.Constant(0.1))
net.collect_params('^FC1_output').initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
net.collect_params('^(?!FC).*$').load(param_file, ignore_extra=True)
result = net(data)


# Export a modifed baseline model with a 256 FC layer & 13003 FC layer output
print("Exporting to: %s \n" % out_path)
net.export(out_path)



