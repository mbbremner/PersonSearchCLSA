
# ============================================================
# --------------------< CLSA Training  >----------------------
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

    identifier = 'x'                # Unique session identifier

    # Default Values
    batch_size = 32                 # Training batch size
    N = 0                           # Number of classes
    epochs = 0                      # Training Epoch limit
    learning_rate = 0.001           # Model learning rate
    outfile = "default_out.txt"     # Save trained model
    print_frequency = 10            # Print every n batches
    T = 1                           # CLSA Temperature

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

    # Input Files
    param_file_bbone, sym_file_bbone = "", ""
    param_file_low, sym_file_low, param_file_med, sym_file_med = "", "", "", ""
    param_file_high, sym_file_high = "", ""

    # Tracking arrays for training
    predicts_tr, labels_tr = [], []

    mean_batch_loss, mean_batch_CE, mean_batch_KL = 0, 0, 0

    group_losses, group_CE_losses, group_KL_losses = [], [], []  # print_frequency tracking
    epoch_losses, epoch_labels, epoch_predictions = [], [], []  # epoch tracking
    epoch_KL_losses, epoch_CE_losses = [], []

    # Aggregate arrays
    accuracies, loss_tuples, loss_deltas, old_losses = [], [], [], (0, 0, 0)

    # -----------------------------------------
    # ------< Constructor / Initialize >-------
    # -----------------------------------------

    def __init__(self):

        # Initialize argparser and define application arguments
        self.ap = argparse.ArgumentParser()
        self.create_argparser()

        # Parse Arguments
        self.opts = self.ap.parse_args()

        # Assign Arguments
        self.T = self.opts.T[0]
        self.batch_size = self.opts.batch_size[0]                   # Batch Size
        self.N = self.opts.N[0]                                     # Number of unique IDs
        self.epochs = self.opts.epochs[0]                           # Number of epochs
        self.learning_rate = self.opts.learning_rate[0]             # Learning Rate
        self.out_path = self.opts.outfile[0]                        # Output path
        self.print_frequency = self.opts.print_frequency[0]         # Print after # of batches
        self.model = self.opts.model[0]                             # Model variant identifier
        self.identifier = self.opts.id[0]                           # Unique session identifier

        context = self.opts.ctx
        print('Context = %s ' % context)
        if context == 'gpu':
            self.ctx = mx.gpu()
        elif context == 'cpu':
            self.ctx = mx.cpu()
        else:
            exit('  >> Invalid Context \n  >> Valid Options: gpu, cpu ')

        # Resnet Backbone Sym / Param paths
        self.param_file_bbone = 'UntrainedModels/ResNetBackbone_' + self.model + '-0000.params'    # Model Parameter file
        self.sym_file_bbone = 'UntrainedModels/ResNetBackbone_' + self.model + '-symbol.json'      # Model architecture file

        self.param_file_low = 'UntrainedModels/CLSA_low' + '-0000.params'    # Model Parameter file
        self.sym_file_low = 'UntrainedModels/CLSA_low' + '-symbol.json'      # Model architecture file

        self.param_file_med = 'UntrainedModels/CLSA_med' + '-0000.params'    # Model Parameter file
        self.sym_file_med = 'UntrainedModels/CLSA_med' + '-symbol.json'      # Model architecture file

        self.param_file_high = 'UntrainedModels/CLSA_high' + '-0000.params'    # Model Parameter file
        self.sym_file_high = 'UntrainedModels/CLSA_high' + '-symbol.json'      # Model architecture file

    # -----------------------------------------
    # ---< Application Arguments Functions >---
    # -----------------------------------------

    def create_argparser(self):

        self.ap.add_argument('-id', nargs=1, type=str, default=['x'])
        self.ap.add_argument('-T', nargs=1, type=int, default=[5])
        self.ap.add_argument('-epochs', nargs=1, type=int, default=[10])
        self.ap.add_argument('-N', nargs=1, type=int, default=[13003])
        self.ap.add_argument('-batch_size', type=int, nargs=1, default=[32])
        self.ap.add_argument('-learning_rate', type=float, nargs=1, default=[0.0001])
        self.ap.add_argument('-outfile', type=str, nargs=1, default=['CLSAModelOut'])
        self.ap.add_argument('-print_frequency', type=int, nargs=1, default=[10])
        self.ap.add_argument('-model', type=str, nargs=1, default=['512'])
        self.ap.add_argument('-ctx', type=str, default=['gpu'])

    # Displays application arguments in terminal
    def print_application_arguments(self):
        print("\n >>>> Application Arguments: \n")
        for opt in vars(self.opts):
            print("   >> " + str(opt) + " " + str(getattr(self.opts, opt)))
        print('\n\n')

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

    # Fetch resnet-50 output layers for CLSA input
    def get_CLSA_inputs(self, model_internals):
        regular_output = [item for item in model_internals if re.search('10_dense', str(item))][0]
        low_lvl = model_internals['resnetv10_stage2_activation3_output']  # low_lvl input
        mid_lvl = model_internals['resnetv10_stage3_activation5_output']  # med_lvl input
        high_lvl = model_internals['resnetv10_stage4_activation2_output']  # high_lvl input
        # regular_output = internals['resnetv10_dense0_fwd_output']
        CLSA_inputs = [regular_output, low_lvl, mid_lvl, high_lvl]
        return CLSA_inputs

    # Load the models using file names from prev. function
    def load_resnet_backbone(self):
        print("This is where to do the loading")
        print(self.model)

        # ---- Load Backbone Resnet-50 Model ---
        sym50 = mx.sym.load(self.sym_file_bbone)  # Load symbol file
        internals = sym50.get_internals()  # Model Internals

        # Get outputs for CLSA inputs
        CLSA_inputs = self.get_CLSA_inputs(internals)

        # Attach outputs to backbone model
        backbone_model = gluon.nn.SymbolBlock(outputs=CLSA_inputs, inputs=mx.sym.var('data'))
        backbone_model.collect_params().load(self.param_file_bbone, ignore_extra=True, ctx=self.ctx)

        return backbone_model

    def load_CLSA_modules(self):
        # -------- Load CLSA Symbol Files ------
        print("Loading CLSA Modules\n")
        sym_low = mx.sym.load(self.sym_file_low)
        sym_med = mx.sym.load(self.sym_file_med)
        sym_high = mx.sym.load(self.sym_file_high)
        low_internals = sym_low.get_internals()
        med_internals = sym_med.get_internals()
        high_internals = sym_high.get_internals()

        # Get proper output layers using re (this is dynamic for switching between models)
        low_layer = [item for item in low_internals if re.search('sigmoid_fwd', str(item))]
        med_layer = [item for item in med_internals if re.search('sigmoid_fwd', str(item))]
        high_layer = [item for item in high_internals if re.search('sigmoid_fwd', str(item))]

        #  ---------- Load CLSA Params ----------
        low_net = gluon.nn.SymbolBlock(outputs=low_layer[0], inputs=mx.sym.var('data'))
        low_net.collect_params().load(self.param_file_low, ignore_extra=True, ctx=self.ctx)

        med_net = gluon.nn.SymbolBlock(outputs=med_layer[0], inputs=mx.sym.var('data'))
        med_net.collect_params().load(self.param_file_med, ignore_extra=True, ctx=self.ctx)

        high_net = gluon.nn.SymbolBlock(outputs=high_layer[0], inputs=mx.sym.var('data'))
        high_net.collect_params().load(self.param_file_high, ignore_extra=True, ctx=self.ctx)

        return low_net, med_net, high_net

    # -----------------------------------------
    # ---------< Training Functions >----------
    # -----------------------------------------
    def execute_training(self):

        # -------------- Loading ---------------
        resnet_backbone = self.load_resnet_backbone()
        CLSA_low_net, CLSA_med_net, CLSA_high_net = self.load_CLSA_modules()

        # ------- Get Training Iterator -------
        self.get_iters(batch_size=self.batch_size)

        # ----------- Init trainers -----------
        trainer1 = gluon.Trainer(resnet_backbone.collect_params(), 'sgd', {'learning_rate': self.learning_rate, 'wd': 0.9})
        trainer2 = gluon.Trainer(CLSA_low_net.collect_params(), 'sgd', {'learning_rate': self.learning_rate, 'wd': 0.9})
        trainer3 = gluon.Trainer(CLSA_med_net.collect_params(), 'sgd', {'learning_rate': self.learning_rate, 'wd': 0.9})
        trainer4 = gluon.Trainer(CLSA_high_net.collect_params(), 'sgd', {'learning_rate': self.learning_rate, 'wd': 0.9})

        # ------- Init. Loss Functions --------
        CE_loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
        KL_loss_function = gluon.loss.KLDivLoss(from_logits=False)

        # -----------------< BEGIN TRAINING >----------------

        for epoch in range(self.epochs):

            print(" >>>> " + "=" * 25 + "" + str("< Epoch # %d >" % (epoch + 1)) + "=" * 25 + '\n')  # Display Epoch #

            # Reset update arrays
            self.train_iter.reset()
            self.reset_epoch_arrays()

            self.group_begin_time = time.monotonic()                             # print_freq begin time
            self.epoch_begin_time = time.monotonic()                             # epoch begin time

            # ---------------< BEGIN EPOCH >---------------
            for batch_id, batch in enumerate(self.train_iter):

                # Check for print_frequency & provide output
                self.print_frequency_output(batch_id)

                # Fetch data & labels from batch data
                self.batch_data = batch.data[0].as_in_context(self.ctx)                # Data: Images
                self.batch_labels = batch.label[0].as_in_context(self.ctx)             # Data: Labels

                # Apply input augmentation
                self.input_augmentation()

                with autograd.record():

                    # Get backbone outputs
                    semantic_vectors = resnet_backbone(self.batch_data)  # Backbone output

                    # Feed backbone outputs into CLSA modules
                    low_lvl_semantics = CLSA_low_net(semantic_vectors[1])  # Low lvl semantic features
                    med_lvl_semantics = CLSA_med_net(semantic_vectors[2])  # Mid lvl semantic features
                    high_lvl_semantics = CLSA_high_net(semantic_vectors[3])  # High lvl semantic features

                    # Apply softening to Low & Med level semantics
                    low_softened = self.apply_softening(low_lvl_semantics)
                    med_softened = self.apply_softening(med_lvl_semantics)

                    # Compute Losses
                    high_mid_loss = KL_loss_function(med_softened, high_lvl_semantics)                      # KL loss high to med
                    high_low_loss = KL_loss_function(low_softened, high_lvl_semantics)                      # KL loss high to low
                    classification_loss = CE_loss_function(high_lvl_semantics, self.batch_labels)           # Cross Entropy loss
                    KL_loss_aggregate = self.T ** 2 * (high_low_loss + high_mid_loss)                       # Total KL Loss
                    total_loss = classification_loss + KL_loss_aggregate                                    # Aggregate loss

                    # Update Tracking
                    self.update_loss_tracking_arrays(total_loss, classification_loss, KL_loss_aggregate)
                    batch_predictions = [np.argmax(item) for item in high_lvl_semantics.asnumpy()]
                    self.predicts_tr.extend(batch_predictions)
                    self.labels_tr.extend(self.batch_labels.asnumpy())

                # Update Parameters
                total_loss.backward()
                trainer1.step(self.batch_data.shape[0], ignore_stale_grad=True)
                trainer2.step(self.batch_data.shape[0], ignore_stale_grad=True)
                trainer3.step(self.batch_data.shape[0], ignore_stale_grad=True)
                trainer4.step(self.batch_data.shape[0], ignore_stale_grad=True)

            # ----------------< END EPOCH >----------------

            # Display epoch summary
            self.print_epoch_summary()

            # Loss Check / Check-pointing
            if self.loss_deltas[0] > 0 and epoch > 0:
                print("\n >>>> Negative loss, break training \n")
                break
            elif epoch > 5:
                self.checkpoint_model(resnet_backbone, CLSA_low_net, CLSA_med_net, CLSA_high_net)

        # ------------------< END TRAINING >-----------------

        self.save_training_summary()

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

    # CLSA Temperature Softening
    def apply_softening(self, semantic_vector):

        soft_top = [np.exp((1 / self.T) * item) for i, item in enumerate(semantic_vector.asnumpy())]
        soft_bottom = [sum(np.exp((1 / self.T) * item.asnumpy())) for item in semantic_vector]

        return mx.nd.array([item / soft_bottom[i] for i, item in enumerate(soft_top)]).as_in_context(self.ctx)

    # Returns epoch accuracy given an array of prediction and ground truth
    def compute_accuracy(self, prediction, ground_truth):
        return len([item for i, item in enumerate(prediction) if int(item) == int(ground_truth[i])]) / len(prediction)

    # Saves the model with the available naming scheme
    def checkpoint_model(self, bbone, low, med, high):

        # Export Trained Models
        print("\n >>>> Check-pointing Model \n")
        low.export('Models/CLSA_' + self.model + '_low_trained' + str(self.batch_size))
        med.export('Models/CLSA_' + self.model + '_med_trained' + str(self.batch_size))
        high.export('Models/CLSA_' + self.model + '_high_trained' + str(self.batch_size))
        bbone.export('Models/CLSA2_' + self.model + '_backbone_trained' + str(self.batch_size))

    # -----------------------------------------
    # -------< Array Update Functions >--------
    # -----------------------------------------

    # Batch-wise updating
    def update_loss_tracking_arrays(self, total_loss, classification_loss, KL_loss):

        self.mean_batch_loss = (sum(total_loss).asnumpy() / len(total_loss))[0]
        self.mean_batch_CE = (sum(classification_loss).asnumpy() / len(classification_loss))[0]
        self.mean_batch_KL = (sum(KL_loss).asnumpy() / len(KL_loss))[0]
        self.group_losses.append(self.mean_batch_loss)
        self.group_KL_losses.append(self.mean_batch_KL)
        self.group_CE_losses.append(self.mean_batch_CE)

    # Reset tracking arrays after every epoch
    def reset_epoch_arrays(self):

        self.group_losses, self.group_CE_losses, self.group_KL_losses = [], [], []  # print_frequency tracking
        self.predicts_tr, self.labels_tr = [], []
        self.epoch_losses, self.epoch_labels, self.epoch_predictions = [], [], []  # epoch tracking
        self.epoch_KL_losses, self.epoch_CE_losses = [], []

    # Reset tracking array after every print_frequency
    def reset_group_arrays(self):
        self.group_losses, self.group_CE_losses, self.group_KL_losses = [], [], []  # print_frequency tracking

    # -----------------------------------------
    # ----------< Display Functions >----------
    # -----------------------------------------

    # Displays this after print_frequency batches, g
    #
    #
    #
    #
    # iven as an argument on the command line
    def print_frequency_output(self, batch_id):
        if batch_id != 0 and batch_id % self.print_frequency == 0:
            group_extime = time.monotonic() - self.group_begin_time
            self.group_begin_time = time.monotonic()
            print_tuple = (batch_id, batch_id + self.print_frequency, group_extime, np.mean(self.group_KL_losses), np.mean(self.group_CE_losses))
            print("   >> Batch #%3d - #%3d -- Time: %5.2f s -- KL Loss: %5.3f   CE Loss: %5.3f " % print_tuple)
            self.epoch_CE_losses.extend(self.group_CE_losses)
            self.epoch_KL_losses.extend(self.group_KL_losses)
            self.epoch_losses.extend(self.group_losses)
            self.group_losses[:] = []
            self.group_CE_losses[:] = []
            self.group_KL_losses[:] = []


    # Displays a summary of vital data every epoch
    def print_epoch_summary(self):

        # Update & Compute
        epoch_mean_losses = (np.mean(self.epoch_losses), np.mean(self.epoch_CE_losses), np.mean(self.epoch_KL_losses))
        self.loss_deltas = [epoch_mean_losses[i] - self.old_losses[i] for i, item in enumerate(epoch_mean_losses)]
        self.loss_tuples.append(epoch_mean_losses)
        epoch_extime = time.monotonic() - self.epoch_begin_time
        epoch_accuracy = self.compute_accuracy(self.predicts_tr, self.labels_tr)

        # Update
        self.old_losses = epoch_mean_losses
        self.accuracies.append(epoch_accuracy)

        # Print epoch summary
        print("\n >>>> Epoch Summary: ")
        print("   >> Accuracy: %5.7f %c \n" % (100 * epoch_accuracy, '%'))
        print("   >> Time: %5.7f mins \n" % (epoch_extime / 60))
        print("   >> Per Image Loss: %6s %+f " % (str(epoch_mean_losses[0]), np.float(self.loss_deltas[0])))
        print("   >> CE Loss:  %5s  %+4.4f" % (str(epoch_mean_losses[1]), np.float(self.loss_deltas[1])))
        print("   >> KL Loss: %6s  %+4.4f" % (str(epoch_mean_losses[2]), np.float(self.loss_deltas[2])))

    # Top Welcome Banner
    def print_welcome(self):
        print(" \n==================================================================================")
        print("                     Cross Level Semantic Alignment Training                       ")
        print(" ==================================================================================\n")

    def save_training_summary(self):
        # Write Training Summary to file
        file_name = 'CLSA_' + self.model + '_training_summary_' + str(self.batch_size) + '.txt'
        f = IO.open(file_name, 'w')
        for i, item in enumerate(self.accuracies):
            f.write(str(item) + ', ' + str(self.loss_tuples[i][0]) + ', ' + str(self.loss_tuples[i][1]) + ', ' + str(self.loss_tuples[i][2]) + '\n')
        f.close()

    # -----------------------------------------
    # ------< Post-Training Functions >--------
    # -----------------------------------------

    def extract_features(self):
        print("Lovely extracted features")

        # Test Iterator

        # Validation Iterator

    def analyze_features(self):
        print("Analysis by Paralysis")


# --------------< Script >----------------

CLSA_Net = MyNeuralNet()
CLSA_Net.print_welcome()
CLSA_Net.print_application_arguments()

# Execute Training Process
CLSA_Net.execute_training()



# ============================================================
# -------------------------< End >----------------------------
# ============================================================


