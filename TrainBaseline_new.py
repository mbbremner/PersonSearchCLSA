
# ============================================================
# ------------< CLSA Feature Pyramid Construction  >----------
# ============================================================

# ----------------- Imports ------------------

import mxnet as mx
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import io as IO
import numpy as np
import argparse
import time
import re
from scipy import misc


# ---------< Primary Class >--------------

class MyNeuralNet:

    NNet_Model = gluon.nn.Sequential()

    # Default Values
    batch_size = 32                 # Training batch size
    N = 0                           # Number of classes
    epochs = 0                      # Training Epoch limit
    learning_rate = 0.001           # Model learning rate
    outfile = "default_out.txt"     # Save trained model
    print_frequency = 10            # Print every n batches
    identifier = 'x'

    group_begin_time = 0            # Begin print_frequency group execution time
    epoch_begin_time = 0            # Begin epoch execution time

    batch_data = []                 # Current batch data
    batch_labels = []               # Current batch labels

    # Execution Context
    cpu_ctx = mx.cpu()              # CPU context
    gpu_ctx = mx.gpu()              # GPU context
    ctx = mx.gpu()

    # Initializations
    ap = []                                         # Argument Parser
    train_iter, test_iter, val_iter = [], [], []    # Data iterators
    opts = []                                       # Application arguments

    # Model Parameter & Symbol files
    param_file_bbone, sym_file_bbone = "", ""

    # Tracking arrays for training
    predicts_tr, labels_tr = [], []

    mean_batch_loss, mean_batch_CE = 0, 0

    group_losses = []                                               # print_frequency tracking
    epoch_losses, epoch_labels, epoch_predictions = [], [], []      # epoch tracking

    epoch_CE_losses, group_CE_losses = [], []
    epoch_KL_losses, group_KL_losses = [], []
    # Aggregate arrays
    accuracies, loss_tuples, loss_delta, old_loss = [], [], 0, 0

    # Feature Stuff
    feature_file = ''

    feature_vector_list = []

    # -----------------------------------------
    # ------< Constructor / Initialize >-------
    # -----------------------------------------

    def __init__(self):

        # Initialize argparser and define application arguments
        self.ap = argparse.ArgumentParser()
        self.create_argparser()

        # Parse Arguments
        self.opts = self.ap.parse_args()

        self.batch_size = self.opts.batch_size[0]            # Batch Size
        self.N = self.opts.N[0]                              # Number of unique IDs
        self.epochs = self.opts.epochs[0]                    # Number of epochs
        self.learning_rate = self.opts.learning_rate[0]      # Learning Rate
        self.print_frequency = self.opts.print_frequency[0]  # Print after # of batches
        self.model = self.opts.model[0]
        self.identifier = self.opts.id[0]

        context = self.opts.ctx
        print('Context = %s ' % context)

        if context == 'gpu':
            self.ctx = mx.gpu()
        elif context == 'cpu':
            self.ctx = mx.cpu()
        else:
            exit('  >> Invalid Context \n  >> Valid Options: gpu, cpu ')

        # Resnet Backbone Sym / Param paths
        self.param_file_bbone = 'UntrainedModels/ResNetBackbone_' + self.model + '-0000.params'  # Model Parameter file
        self.sym_file_bbone = 'UntrainedModels/ResNetBackbone_' + self.model + '-symbol.json'  # Model architecture file

        self.feature_file = 'Features/ResNetBackbone_' + self.identifier

    # -----------------------------------------
    # ---< Application Arguments Functions >---
    # -----------------------------------------

    def create_argparser(self):
        self.ap.add_argument('-id', nargs=1, type=str, default=['x'])
        self.ap.add_argument('-epochs', nargs=1, type=int, default=[10])
        self.ap.add_argument('-N', nargs=1, type=int, default=[13003])
        self.ap.add_argument('-batch_size', type=int, nargs=1, default=[32])
        self.ap.add_argument('-learning_rate', type=float, nargs=1, default=[0.0001])
        self.ap.add_argument('-print_frequency', type=int, nargs=1, default=[10])
        self.ap.add_argument('-model', type=str, nargs=1, default=['512'])
        self.ap.add_argument('-ctx', type=str, default=['gpu'])

    # Displays application arguments in terminal
    def print_application_arguments(self):
        print("\n >>>> Application Arguments: \n")
        for opt in vars(self.opts):
            print("   >> " + str(opt) + " " + str(getattr(self.opts, opt)))
        print('\n')

    # -----------------------------------------
    # ----------< Loading Functions >----------
    # -----------------------------------------

    # Get data iterators function
    # Note this version returns a list to facilitate indexing
    def get_iters(self, batch_size):

        self.train_iter = mx.image.ImageIter(
            batch_size=batch_size,
            data_shape=(3, 128, 256),
            label_width=1,
            path_imgrec='Data/train.rec',
            path_imglist='Data/train.lst'
        )

        self.test_iter = mx.image.ImageIter(
            batch_size=batch_size,
            data_shape=(3, 128, 256),
            label_width=1,
            path_imgrec='Data/test.rec',
            path_imglist='Data/test.lst'
        )

        self.val_iter = mx.image.ImageIter(
            batch_size=batch_size,
            data_shape=(3, 128, 256),
            label_width=1,
            path_imgrec='Data/val.rec',
            path_imglist='Data/val.lst'
        )

    # Load the models using file names from prev. function
    def load_resnet_backbone(self):

        # Load pre-trained Model
        output_layer = 'FC1_output'
        sym = mx.sym.load(self.sym_file_bbone)                                                          # Load symbol file
        feature_layer = sym.get_internals()[output_layer]                                               # Feature layer symbol
        backbone_model = gluon.nn.SymbolBlock(outputs=feature_layer, inputs=mx.sym.var('data'))         # Initialize net w/ desired output
        backbone_model.collect_params().load(self.param_file_bbone, ctx=self.ctx, ignore_extra=True)    # Load model parameters

        return backbone_model

    # Forms a baseline model by loading a pretrained res50 and modifying
    def construct_baseline_model(self):

        out_path = 'TrainedModels/default_name'
        FC_len = 512

        split_pattern = "[ <>]"  # splits model internals for console output

        # Obtain feed-forward data
        self.train_iter.reset()
        for b, batch in enumerate(self.train_iter):

            data = batch.data[0].as_in_context(self.cpu_ctx)
            break

        # Transfer features from here
        resnet50 = vision.resnet50_v1(pretrained=True, ctx=self.cpu_ctx)  # Load pre-trained model
        resnet50.hybridize()

        # Transfer pre-trained features to a new net
        net = vision.resnet50_v1(classes=self.N, prefix='resnetv10_')
        with net.name_scope():
            net.output = nn.Dense(FC_len, flatten=True)
            net.output.collect_params().initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
        net.features = resnet50.features
        net.hybridize()
        net(data)
        net.export('UntrainedModels/Res50_pretrained_model')

        # Load Resnet 50 Model
        sym50 = mx.sym.load(self.sym_file_bbone)  # Load symbol file
        feature_layer = 'FC1_output'  # Desired output layer
        sym_layer = mx.symbol.FullyConnected(name='FC1', num_hidden=self.N)  # Define dense layer
        composed = sym_layer(FC1_data=sym50, name='FC1')  # Combine with loaded symbol model
        internals = composed.get_internals()  # Fetch internals, for verification

        # Bring together network and feed forward
        net = gluon.nn.SymbolBlock(outputs=internals[feature_layer], inputs=mx.sym.var('data'))
        net.collect_params('^FC').initialize(mx.init.Constant(0.1))
        net.collect_params('^FC1_output').initialize(mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2))
        net.collect_params('^(?!FC).*$').load(self.param_file_bbone, ignore_extra=True)
        result = net(data)

        # Export a modifed baseline model with a 256 FC layer & 13003 FC layer output
        print("Exporting to: %s \n" % out_path)
        net.export(out_path)

    # -----------------------------------------
    # ---------< Training Functions >----------
    # -----------------------------------------

    def execute_training(self):

        resnet_backbone = self.load_resnet_backbone()                           # Load Model
        self.get_iters(batch_size=self.batch_size)                              # Get Training Iterator
        loss_function_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()      # Loss function

        # Init trainer
        baseline_trainer = gluon.Trainer(resnet_backbone.collect_params(), 'sgd', {'learning_rate': self.learning_rate, 'wd': 0.9})

        # ======================================================
        # =================< BEGIN TRAINING > ==================
        # ======================================================
        for epoch in range(self.epochs):

            # Console Update
            self.print_epoch_banner(epoch)

            # Reset Tracking Arrays
            self.train_iter.reset()
            self.reset_epoch_arrays()
            # Activate Timers
            self.group_begin_time = time.monotonic()
            self.epoch_begin_time = time.monotonic()

            # ---------------< BEGIN EPOCH >---------------
            for batch_id, batch in enumerate(self.train_iter):

                # Check for periodic output
                self.print_frequency_output(batch_id)

                # Fetch batch data & labels from batch data
                self.batch_data = batch.data[0].as_in_context(self.ctx)
                self.batch_labels = batch.label[0].as_in_context(self.ctx)

                # Apply input augmentation
                self.input_augmentation()

                with autograd.record():

                    classification_output = resnet_backbone(self.batch_data)                                       # Feed Data
                    prediction = nd.softmax(classification_output).asnumpy()                                       # Transform to prediction
                    classification_loss = loss_function_cross_entropy(classification_output, self.batch_labels)    # Cross Entropy loss

                    # Update Tracking
                    self.update_loss_tracking_arrays(classification_loss)
                    batch_predictions = [np.argmax(item) for item in prediction.asnumpy()]
                    self.predicts_tr.extend(batch_predictions)
                    self.labels_tr.extend(self.batch_labels.asnumpy())

                # Update Parameters
                classification_loss.backward()
                baseline_trainer.step(self.batch_data.shape[0])
            # ----------------< END EPOCH >----------------

            # Display epoch summary
            self.print_epoch_summary()

            # Loss Check / Check-pointing
            if self.loss_delta > 0 and epoch > 0:
                print("\n >>>> Negative loss, break training \n")
                break
            elif epoch > 5:
                self.checkpoint_model(resnet_backbone)
                self.NNet_Model = resnet_backbone
        # ===================< END TRAINING >===================

        self.save_training_summary()
        self.NNet_Model = resnet_backbone

    # Input Augmentation, calls transform
    def input_augmentation(self):

        self.batch_data = [self.transform(mx.nd.array(item, dtype="float32")) for item in self.batch_data]
        new_data = self.batch_data[0]
        for item in self.batch_data[1:]:
            new_data = nd.concat(new_data, item, dim=0)
        self.batch_data = new_data
        self.batch_data = self.batch_data.as_in_context(self.ctx)

    # Input Transform
    def transform(self, data):
        # data = data.transpose((2,0,1)).expand_dims(axis=0)
        rgb_mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        rgb_std = mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        # data.astype('float32')
        return (data / 255 - rgb_mean) / rgb_std

    # Returns epoch accuracy given an array of prediction and ground truth
    def compute_accuracy(self, prediction, ground_truth):
        return len([item for i, item in enumerate(prediction) if int(item) == int(ground_truth[i])]) / len(prediction)

    # Saves the model with the available naming scheme
    def checkpoint_model(self, bbone):

        # Export Trained Models
        print("\n >>>> Check-pointing Model \n")
        bbone.export('Models/Baseline_' + self.model + '_bb_trained_' + str(self.batch_size) + '_' + str(self.identifier))

    # -----------------------------------------
    # -------< Array Update Functions >--------
    # -----------------------------------------

    # Batch-wise updating
    def update_loss_tracking_arrays(self, classification_loss):

        self.mean_batch_loss = (sum(classification_loss).asnumpy() / len(classification_loss))[0]
        self.group_losses.append(self.mean_batch_loss)

    # Reset tracking arrays after every epoch
    def reset_epoch_arrays(self):

        self.group_losses, self.group_CE_losses, self.group_KL_losses = [], [], []  # print_frequency tracking
        self.predicts_tr, self.labels_tr = [], []
        self.epoch_losses, self.epoch_labels, self.epoch_predictions = [], [], []  # epoch tracking
        self.epoch_CE_losses = []

    # Reset tracking array after every print_frequency
    def reset_group_arrays(self):
        self.group_losses[:], self.group_CE_losses[:] = [], []  # print_frequency tracking

    # -----------------------------------------
    # ----------< Display Functions >----------
    # -----------------------------------------

    # Top Welcome Banner
    def print_welcome(self):
        print(" \n==================================================================================")
        print("                           Welcome to Baseline Training                              ")
        print(" ==================================================================================\n")

    def print_epoch_banner(self, epoch):
        print(" >>>> " + "=" * 25 + "" + str("< Epoch # %d >" % (epoch + 1)) + "=" * 25 + '\n')

    # Displays this after print_frequency batches, given as an argument on the command line
    def print_frequency_output(self, batch_id):

        if batch_id != 0 and batch_id % self.print_frequency == 0:
            group_extime = time.monotonic() - self.group_begin_time
            self.group_begin_time = time.monotonic()
            print_tuple = (batch_id - self.print_frequency, batch_id, group_extime, np.sum(self.group_losses))
            print("   >> Batch #%3d - #%3d -- Time: %5.2f s -- Loss Sum: %5.2f  " % print_tuple)
            self.epoch_losses.extend(self.group_losses)
            self.group_losses[:] = []

    # Displays a summary of vital data every epoch
    def print_epoch_summary(self):

        # Update & Compute
        epoch_mean_loss = np.mean(self.epoch_losses)
        self.loss_delta = epoch_mean_loss - self.old_loss

        self.loss_tuples.append(epoch_mean_loss)
        epoch_extime = time.monotonic() - self.epoch_begin_time
        epoch_accuracy = self.compute_accuracy(self.predicts_tr, self.labels_tr)

        # Update
        self.old_loss = epoch_mean_loss
        self.accuracies.append(epoch_accuracy)

        # Print epoch summary
        print("\n >>>> Epoch Summary: ")
        print("   >> Accuracy: %5.7f %c \n" % (100 * epoch_accuracy, '%'))
        print("   >> Time: %5.7f mins \n" % (epoch_extime / 60))
        print("   >> Per Image Loss: %6s %+f " % (str(epoch_mean_loss), np.float(self.loss_delta)))

    def save_training_summary(self):
        # Write Training Summary to file
        file_name = 'Baseline_' + self.model + '_training_summary_' + str(self.batch_size) + '.txt'
        f = IO.open(file_name, 'w')
        for i, item in enumerate(self.accuracies):
            f.write(str(item) + ', ' + str(self.loss_tuples[i]) + '\n')
        f.close()

    # -----------------------------------------
    # ------< Post-Training Functions >--------
    # -----------------------------------------

    def extract_features(self, iter):
        print("Lovely extracted features")

        data_iterator = []
        if iter == 'test':
            data_iterator = self.test_iter

        elif iter == 'val':
            data_iterator = self.val_iter
        else:
            exit("bad iter selection")

        # Test Iterator
        feature_vector_list = []
        # ----------- Extraction Loop ----------
        for epoch in range(1):

            print("\n >>>> Processing Images")
            all_labels = []

            for batch_id, batch in enumerate(data_iterator):

                data = batch.data[0].as_in_context(self.ctx)  # Fetch batch image data
                labels = batch.label[0].as_in_context(self.ctx)  # Fetch ground truth labels

                out = self.NNet_Model(data)
                output = nd.flatten(out).asnumpy().tolist()  # Obtain features
                all_labels.extend(labels.asnumpy())  # Extend label list

                for single_output_item in output:  # Append output features to feature list
                    feature_vector_list.append(single_output_item)

                # Display check
                if batch_id % 5 == 0:
                    print("   >> Images Processed: " + str(batch_id * self.batch_size))

        write_features_to_file(feature_vector_list, all_labels, self.feature_file)

        # Validation Iterator

    def analyze_features(self):
        print("Analysis by Paralysis")


# --------------< Script >----------------

Baseline_Net = MyNeuralNet()
Baseline_Net.print_welcome()
Baseline_Net.print_application_arguments()

# Execute Training Process
Baseline_Net.execute_training()



# ============================================================
# -------------------------< End >----------------------------
# ============================================================


