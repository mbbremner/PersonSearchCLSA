# ====================================================================
# ------ Run Baseline Model & Extract list of Identity Features ------
# ====================================================================

#  This scripts:
#
#   (1) Loads a pre-trained gluon model from exported model files
#   (2) Runs the given data set (train or val) through the network
#       to obtain identity semantic features
#   (3) Maintains a list of features + their corresponding tags
#       & labels, and prints them to a .txt file

import mxnet as mx
from mxnet import gluon
import time
import mxnet.ndarray as nd
import io as IO
import argparse
import re
import numpy as np

# --------------------------------------------------------------------
# -------------------------< Functions >------------------------------
# --------------------------------------------------------------------

# Get data iterators function
# Note this version returns a list to facilitate indexing
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

    return [train_iter, test_iter, val_iter]


# Input Transform
def transform(data):

    # data = data.transpose((2,0,1)).expand_dims(axis=0)
    rgb_mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    rgb_std = mx.nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
    # data.astype('float32')
    return (data / 255 - rgb_mean) / rgb_std


# Initializes scripts argparser to accept command line arguments
def create_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('-N', nargs=1, type=int, default=[13003])
    ap.add_argument('-batch_size', type=int, nargs=1, default=[1])
    ap.add_argument('-featurefile', type=str, nargs=1, default=['FeaturesDefault.txt'])
    ap.add_argument('-centroidfile', type=str, nargs=1, default=['CentroidsDefault.txt'])
    ap.add_argument('-sym_in', type=str, nargs=1, default=['Models/Res50_Baseline_Modified-0000.params'])
    ap.add_argument('-param_in', type=str, nargs=1, default=['Models/Res50_Baseline_Modified-symbol.json'])
    ap.add_argument('-iterator', type=int, nargs=1, default=[1])

    return ap


# Saves output to file, given a list of feature vectors
def write_features_to_file(features, labels_in, file_name):
    print("   >> Writing %d Feature Vectors to File: %s" % (len(features), file_name))
    f = IO.open(file_name, 'w')
    for v, vector in enumerate(features):
        out = str(int(labels_in[v])) + '  ,  ' + ','.join([str(item) for item in vector]) + '\n'
        f.write(out)
    f.close()
    print("   >> Done")


# Simple display of arguments as given
def print_application_arguments(options):
    print("\n Application Arguments: \n")
    for opt in vars(options):
        print("   - " + str(opt) + " " + str(getattr(options, opt)))


# Parses the list of identity features and finds the closest
# (or list of closest matches)
def compute_feature_distance(probe_features, feature_list):
    current_closest = -1

    for vector in feature_list:
        print("Placeholder")
        # Compute the norm & compare to current highest
        # if new_norm < current_closest:
        #   current_closest = new_norm


# Welcome Banner display
def print_welcome():
    print("\n\n ==================================================================================")
    print(" ---------------------< Welcome to ResNet Feature Extractor >---------------------")
    print(" ==================================================================================")


# --------------------------------------------------------------------
# ---------------------------< Script >-------------------------------
# --------------------------------------------------------------------

print_welcome()
ctx = mx.gpu()

argparser = create_argparser()                          # Initialize argument parser
opts = argparser.parse_args()                           # Parse input arguments
print_application_arguments(opts)


# Resnet Backbone file
param_file = 'Models/CLSA2_512_backbone_trained32-0000.params'          # Model Parameter file
sym_file = 'Models/CLSA2_512_backbone_trained32-symbol.json'            # Model architecture file
feature_layer = 'resnetv10_dense0_fwd_output'                      # Desired output layer

param_file_low = 'Models/CLSA_512_low_trained32-0000.params'
sym_file_low = 'Models/CLSA_512_low_trained32-symbol.json'
param_file_med = 'Models/CLSA_512_med_trained32-0000.params'
sym_file_med = 'Models/CLSA_512_med_trained32-symbol.json'
param_file_high = 'Models/CLSA_512_high_trained32-0000.params'
sym_file_high = 'Models/CLSA_512_high_trained32-symbol.json'

# Application Arguments
N = opts.N[0]
batch_size = opts.batch_size[0]
selected_iterator = opts.iterator[0]                # Iterator selection, 0::train, 1::test, 2::val
feature_file = opts.featurefile[0]

# ---- Load Backbone Resnet-50 Model ---
sym50 = mx.sym.load(sym_file)                                         # Load symbol file
internals = sym50.get_internals()
low_lvl = internals['resnetv10_stage2_activation3_output']
mid_lvl = internals['resnetv10_stage3_activation5_output']
high_lvl = internals['resnetv10_stage4_activation2_output']
semantic_vector_syms = [low_lvl, mid_lvl, high_lvl]

Resnet50 = gluon.nn.SymbolBlock(outputs=semantic_vector_syms, inputs=mx.sym.var('data'))
Resnet50.collect_params().load(param_file, ignore_extra=True, ctx=ctx)


# -------- Load CLSA Symbol Files ------
print("Loading CLSA Modules\n")
sym_low = mx.sym.load(sym_file_low)
sym_med = mx.sym.load(sym_file_med)
sym_high = mx.sym.load(sym_file_high)
low_internals = sym_low.get_internals()
med_internals = sym_med.get_internals()
high_internals = sym_high.get_internals()

low_out = [item for item in low_internals if re.search('dense', str(item))]
med_out = [item for item in med_internals if re.search('dense', str(item))]
high_out = [item for item in high_internals if re.search('sigmoid_fwd', str(item))]

# ---------- Load CLSA Params ----------
low_net = gluon.nn.SymbolBlock(outputs=low_out[5], inputs=mx.sym.var('data'))
low_net.collect_params().load(param_file_low, ignore_extra=True, ctx=ctx)

med_net = gluon.nn.SymbolBlock(outputs=med_out[5], inputs=mx.sym.var('data'))
med_net.collect_params().load(param_file_med, ignore_extra=True, ctx=ctx)

high_net = gluon.nn.SymbolBlock(outputs=high_out, inputs=mx.sym.var('data'))
high_net.collect_params().load(param_file_high, ignore_extra=True, ctx=ctx)

iterators = get_iters(batch_size=batch_size)
iterator = iterators[selected_iterator]


all_features = []
all_labels = []
# ----------- Extraction Loop ----------
for epoch in range(1):

    print(" >>>> " + "=" * 25 + "" + str("< Epoch # %d >" % (epoch + 1)) + "=" * 25 + '\n')  # Display Epoch #

    iterator.reset()

    group_begin = time.monotonic()
    epoch_begin_time = time.monotonic()

    for batch_id, batch in enumerate(iterator):

        data = batch.data[0].as_in_context(ctx)                 # Data: Images
        labels = batch.label[0].as_in_context(ctx)              # Data: Labels

        all_labels.extend(labels.asnumpy())

        semantic_vectors = Resnet50(data)

        out1 = low_net(semantic_vectors[0])
        out2 = med_net(semantic_vectors[1])
        out3 = high_net(semantic_vectors[2])
        combined = nd.concat(out1, out2, out3).asnumpy()
        print( " >> %d" % len(combined))
        # print(len(out3))
        for row in combined:
            all_features.append(row)


# print(len(all_features))
# for row in all_features:
#     print(len(row))
#
write_features_to_file(all_features, all_labels, feature_file)


# --------------------------------------------------------------------
# -----------------------------< End >--------------------------------
# --------------------------------------------------------------------
