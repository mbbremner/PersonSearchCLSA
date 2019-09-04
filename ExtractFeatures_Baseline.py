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
import mxnet.ndarray as nd
import io as IO
import argparse
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
    ap.add_argument('-epochs', nargs=1, type=int, default=[1])
    ap.add_argument('-N', nargs=1, type=int, default=[13003])
    ap.add_argument('-batch_size', type=int, nargs=1, default=[1])
    ap.add_argument('-featurefile', type=str, nargs=1, default=['FeaturesDefault.txt'])
    ap.add_argument('-iterator', type=int, nargs=1, default=[1])
    ap.add_argument('-sym_in', type=str, nargs=1, default=['Models/Res50_Baseline_Modified-0000.params'])
    ap.add_argument('-param_in', type=str, nargs=1, default=['Models/Res50_Baseline_Modified-symbol.json'])

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

# Argparser initialization
argparser = create_argparser()                      # Initialize argument parser
opts = argparser.parse_args()                       # Parse input arguments
print_application_arguments(opts)                   # Display Args

# Arguments
feature_file = opts.featurefile[0]                  # Save features to this file
selected_iterator = opts.iterator[0]                # Iterator selection, 0::train, 1::test, 2::val
N = opts.N[0]                                       # Not Important
epochs = opts.epochs[0]                             # Should be 1 for this scripts
batch_size = opts.batch_size[0]                     # Model & Data context
ctx = mx.gpu()
# ctx = mx.cpu()
# Trained model files
param_file = opts.param_in[0]
sym_file = opts.sym_in[0]
feature_layer = 'resnetv10_dense1_fwd_output'                      # Desired output layer
# param_file = 'Models/Res50_Baseline_Modified-0000.params'        # Model Parameter file
# sym_file = 'Models/Res50_Baseline_Modified-symbol.json'          # Model architecture file


# Load trained model
sym = mx.sym.load(sym_file)                                                     # Load symbol file
internals = sym.get_internals()                                                 # Model Internals
feature_layer = internals[feature_layer]                                        # Feature layer symbol
net = gluon.nn.SymbolBlock(outputs=feature_layer, inputs=mx.sym.var('data'))    # Initialize net w/ desired output
net.collect_params().load(param_file, ctx=ctx, ignore_extra=True)               # Load model parameters


iterators = get_iters(batch_size)                                               # Fetch iterators
input_iterator = iterators[selected_iterator]                                   # Train, Test, or Val, default = Test
feature_vector_list = []                                                        # Maintains each obtained feature vector

# ----------- Extraction Loop ----------
for epoch in range(1):

   print("\n >>>> Processing Images")
    input_iterator.reset()
    all_labels = []

    for batch_id, batch in enumerate(input_iterator):

        data = batch.data[0].as_in_context(ctx)                                 # Fetch batch image data
        labels = batch.label[0].as_in_context(ctx)                              # Fetch ground truth labels

        out = net(data)
        output = nd.flatten(out).asnumpy().tolist()                             # Obtain features
        all_labels.extend(labels.asnumpy())                                     # Extend label list

        for single_output_item in output:                                       # Append output features to feature list
            feature_vector_list.append(single_output_item)

        # Display check
        if batch_id % 5 == 0:
            print("   >> Images Processed: " + str(batch_id * batch_size))

# ----------<Write Features to File >----------
write_features_to_file(feature_vector_list, all_labels, feature_file)

print("\n\n >>>>>> Thanks for Extracting Features\n\n")


# # ----------< Compute Class Centroids >----------
# initial_class = all_labels[0]
# label_count = 0
# class_centroids = []
# class_vectors = []
# old_label = 0
#
# for r, row in enumerate(feature_vector_list):
#
#     print(row)
#
#     label = int(all_labels[r])
#     # print("Old: %d  |  New: %d" % (old_label, label))
#
#     if label == initial_class:
#         label_count += 1
#         class_vectors.append(row)
#     else:
#
#         # Compute Mean of Class Vectors
#         centroid = np.mean(class_vectors, axis=0)
#         class_centroids.append(centroid)
#         #
#         # print("   >> Label ID %d, Total Count = %d" % (label, label_count))
#         # print("Class Centroid: %s" % str(centroid[0:5]))
#         label_count = 1
#         class_vectors = []
#         class_vectors.append(row)
#         # print(" >>>> New Label %d, count = %d" % (label, label_count))
#         initial_class = all_labels[r]
#
#         if label < old_label:
#             print("\n   >> End of Data, Truncating Batch Overrun & Breaking")
#             break
#
#     old_label = label
#
#
# # ----------< Write Class Centroids >----------
# print("\n >>>> Saving Features to .txt Files \n")
# labels = list(set(all_labels))
# f = IO.open(centroid_file, 'w')
#
# for c, centroid in enumerate(class_centroids):
#     f.write(str(labels[c]) + ','.join([str(item) for item in centroid]))
#
# f.close()



# write_features_to_file(class_centroids, labels, centroid_file)

