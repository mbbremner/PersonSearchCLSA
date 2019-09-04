# ============================================================
# ------------< CLSA Feature Pyramid Construction  >----------
# ============================================================

# The purpose of this script is to generate and save the 3 CLSA
# feature pyramid modules

import mxnet as mx
from mxnet import gluon
import argparse
import re


# --------------------------------------------------------------------
# -------------------------< Functions >------------------------------
# --------------------------------------------------------------------

# Application specific argument parser
def create_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('-N', nargs=1, type=int, default=[13003])
    # ap.add_argument('-outfile', type=str, nargs=1, default=['Models/Res50_Baseline_Modified'])
    ap.add_argument('-sym_in', type=str, nargs=1, default=['Models/Res50_pretrained_model-0000.params'])
    ap.add_argument('-param_in', type=str, nargs=1, default=['Models/Res50_pretrained_model-symbol.json'])
    ap.add_argument('-model', nargs=1, type=str, default=['512'])
    ap.add_argument('-dense_size', type=int, nargs=1, default=[256])
    return ap

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


# Initialize CLSA module
def init_CLSA_module(N, v):

    net = gluon.nn.HybridSequential()
    net.add(gluon.nn.GlobalAvgPool2D(layout='NCHW'))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(v))
    net.add(gluon.nn.BatchNorm())
    net.add(gluon.nn.Activation('relu'))
    net.add(gluon.nn.Dense(v))
    net.add(gluon.nn.Dense(N, activation='sigmoid'))
    # net.add(gluon.nn.LayerNorm())
    net.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
    return net


# Takes argparser opts as input, displays application arguments in terminal
def print_application_arguments(options):
    print("\n Application Arguments: \n")
    for opt in vars(options):
        print("   - " + str(opt) + " " + str(getattr(options, opt)))
    print('\n\n')


# Top Welcome Banner
def print_welcome():
    print(" ==================================================================================")
    print("                            CLSA Module Construction                               ")
    print(" ==================================================================================\n")


# --------------------------------------------------------------------
# ---------------------------< Script >-------------------------------
# --------------------------------------------------------------------

print_welcome()                                 # Welcome banner upon execution
argparser = create_argparser()                  # Initialize argument parser
opts = argparser.parse_args()                   # Parse input arguments
print_application_arguments(opts)               # Display input arguments



FC_length = opts.dense_size[0]
N = opts.N[0]
model = opts.model[0]

param_file = 'Models/ResNetBackbone_' + model + '-0000.params'            # Model Parameter file
sym_file = 'Models/ResNetBackbone_' + model + '-symbol.json'              # Model architecture file
feature_layer = 'resnetv10_dense0_fwd_output'                   # Desired output layer

low_layer = 'resnetv10_stage2_activation3_output'               # Target layer to extract low lvl features
mid_layer = 'resnetv10_stage3_activation5_output'               # Target layer to extract low lvl features
high_layer = 'resnetv10_stage4_activation2_output'              # Target layer to extract high  lvl features


# Load Resnet 50 Model
sym50 = mx.sym.load(sym_file)                                         # Load symbol file
internals = sym50.get_internals()
feature_layer = [item for item in internals if re.search('10_dense', str(item))][0]


regular_output = feature_layer
low_lvl = internals[low_layer]
mid_lvl = internals[mid_layer]
high_lvl = internals[high_layer]
semantic_vector_syms = [regular_output, low_lvl, mid_lvl, high_lvl]

Resnet50 = gluon.nn.SymbolBlock(outputs=semantic_vector_syms, inputs=mx.sym.var('data'))
Resnet50.collect_params().load(param_file, ignore_extra=True)


print(" >>>> Our Lovely Output Layers:\n")
print("   >> Low level : %s" % str(low_lvl))
print("   >> mid level : %s" % str(mid_lvl))
print("   >> high level : %s" % str(high_lvl))

print(" ------ \n")


# Initialize CLSA Modules
low_net = init_CLSA_module(N, FC_length)
med_net = init_CLSA_module(N, FC_length)
high_net = init_CLSA_module(N, FC_length)

low_net.hybridize(), med_net.hybridize(), high_net.hybridize()


print(" Nets succesfully hybridized")
# Step forward to get module parameters for export
ctx = mx.cpu()
train_iter, _, _ = get_iters(batch_size=32)
for b, batch in enumerate(train_iter):

    data = batch.data[0].as_in_context(ctx)
    labels = batch.label[0].as_in_context(ctx)
    semantic_vectors = Resnet50(data)

    output0 = semantic_vectors[0]
    output1 = low_net(semantic_vectors[1])
    output2 = med_net(semantic_vectors[2])
    output3 = high_net(semantic_vectors[3])
    break


# Export
out_paths = [str('/'.join(('Models', item)) + '_' + str(FC_length)) for item in ('CLSA_low', "CLSA_med", "CLSA_high")]
print("Exporting models to: %s, %s, %s " % (out_paths[0], out_paths[1], out_paths[2]))
low_net.export(out_paths[0])
med_net.export(out_paths[1])
high_net.export(out_paths[2])



