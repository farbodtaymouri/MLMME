from __future__ import print_function, division
import os
import sys
import torch
# import pandas as pd
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# from torchvision import transforms, utils
# import torch.nn as nn
# from scipy.special import softmax
# import torchvision
# from torch.autograd import Variable
# from sklearn.decomposition import PCA
# import seaborn as sns
from tqdm import tqdm
# from sklearn.metrics import accuracy_score
# from sklearn.manifold import TSNE
# import torchvision
# import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=1000)
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
# import torch.nn.init  as init
# import pandas as pd
# import random
# import pprint
# import copy
import random
# from torch.nn.utils.rnn import pad_sequence
# from sklearn.metrics import precision_recall_fscore_support
# from collections import defaultdict
# import collections
from similarity.damerau import Damerau
# import xlsxwriter
# import pickle
# import pathlib
import preparation as pr
# device=torch.device('cuda:0')
# plt.style.use('ggplot')


class Encoder(nn.Module, pr.Preprocessing):
    def __init__(self, input_size, batch, hidden_size, num_layers, num_directions):
        nn.Module.__init__(self)
        #pr.Preprocessing.__init__(self)
        # self.h = torch.zeros(num_layers * num_directions,batch, hidden_size).cuda()
        # self.c = torch.zeros(num_layers * num_directions,batch, hidden_size).cuda()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True, bidirectional=False)
        self.hid_dim = hidden_size
        self.n_layers = num_layers

        # Define sigmoid activation and softmax output
        # self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        # output, (h,c) = self.lstm(x, (self.h.detach(), self.c.detach()))
        self.lstm.flatten_parameters()  # For running DataParallel
        output, (h, c) = self.lstm(x)

        return h, c

#
# seq_len = 1  # (not important)
# # input_size = len(unique_event)+1  #At the moment only considering the events + '0' for the end of trace
# input_size = len(selected_columns)
# # input_size = len(selected_columns)-1 #Ignoring the end of statement
# batch = batch
# # hidden_size= input_size*2
# hidden_size = 200
# num_layers = 5
# num_directions = 1  # It should be 2 if we use bidirectional

# enc = Encoder( input_size, batch, hidden_size, num_layers, num_directions)
# enc.cuda()


# optimizerG = torch.optim.Adam(enc.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00002)
# print(enc)


#############################################################

class Decoder(nn.Module, pr.Preprocessing):
    def __init__(self, input_size, batch, hid_dim, n_layers, dropout):
        super().__init__()

        # self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = input_size

        self.rnn = nn.LSTM(input_size, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, input_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.fc_dropout = nn.Dropout()
        # self.dropout = nn.Dropout(dropout)


    def forward(self, input, hidden, cell):
        duration_time_loc = self.duration_time_loc
        self.rnn.flatten_parameters()  # For running DataParallel for multiple GPU (otherwise you can disable it)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))

        # prediction = self.relu(self.fc_dropout(self.fc_out(output)))
        prediction = self.relu(self.fc_out(output))
        # prediction = [batch size, output dim]

        # return prediction, hidden, cell
        prediction[:, :, duration_time_loc] = self.relu(prediction[:, :, duration_time_loc].clone())
        return prediction, hidden, cell


# dec = Decoder(input_size, batch, hidden_size, num_layers, dropout=.3)
# dec.cuda()


# optimizerG = torch.optim.Adam(dec.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00002)
# print(dec)

# #######################################################################################
class Seq2Seq(nn.Module, pr.Preprocessing):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        hidden, cell = self.encoder(src)

        # trg = trg[:,:,selected_columns]
        # print(trg.size(), begin_sentence.size())
        begin_symbol = torch.ones((trg.size()[0], 1, trg.size()[2])).cuda()
        inp = begin_symbol

        # Iterating over the length of suffix
        for i in range(trg.size()[1]):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            # print(output.size())

            # decide if we are going to use teacher forcing or not
            # teacher_forcing_ratio =0.5
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                inp = trg[:, i, :].view((trg.size()[0], 1, trg.size()[2]))
            else:
                # inp = torch.nn.functional.gumbel_softmax(output, tau = 0.001)
                inp = output

            begin_symbol = torch.cat((begin_symbol, output), dim=1)
        # print("----------------------------", begin_symbol.size())
        prediction = begin_symbol[:, 1:, :]

        return prediction

        # return outputs


# model = Seq2Seq(enc, dec).cuda()
# optimizerG = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00002)

#############################################################################################
class Discriminator(nn.Module, pr.Preprocessing):
    def __init__(self, input_size, batch, hid_dim, n_layers, dropout):
        super().__init__()

        # self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.output_dim = input_size

        self.rnn = nn.LSTM(input_size, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim, 1)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        self.rnn.flatten_parameters()  # For running DataParallel for multiple GPU (otherwise you can disable it)
        output, (hidden, cell) = self.rnn(input)

        prediction = self.fc_out(output)

        return prediction


#############################################################################################
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

#############################################################################################

def train_mle(model,optimizerG, data_obj):
    # Traning seq to seq model using MLE
    # Training Generator (NEW GAME generating suffixes during traning)

    unique_event = data_obj.unique_event
    train_suffix_loader_partition_list = data_obj.train_suffix_loader_partition_list
    selected_columns = data_obj.selected_columns
    duration_time_loc = data_obj.duration_time_loc
    weights_final = data_obj.weights_final



    model.train()
    epoch = 500
    events = list(np.arange(0, len(unique_event) + 1))
    # evetns = list(np.arange(1,len(unique_event)+1))
    gen_loss_pred = []
    disc_loss_tot = []
    gen_loss_tot = []
    gen_loss_temp = []
    disc_loss_temp = []
    accuracy_best = 0
    accuracy_on_validation = []
    loss_on_validation = []
    dl_on_validation = []

    for i in tqdm(range(epoch)):
        # train_suffix_loader = DataLoader(dataset=train_suffix_data, batch_size=batch, shuffle=True)
        for tl in train_suffix_loader_partition_list:
            train_suffix_loader = tl
            # MAX_ITER = tl.dataset[0][1].size()[0]
            # MAX_ITER = tl.dataset.tensors[1].size()[1]
            for mini_batch in iter(train_suffix_loader):
                optimizerG.zero_grad()

                x = mini_batch[0];
                y_truth = mini_batch[1];
                case_id = mini_batch[2]

                y_pred = model(src=x[:, :, selected_columns], trg=y_truth[:, :, selected_columns], teacher_forcing_ratio=0.1)

                # ------------------
                # normal traning (MLE)
                y_truth_label = torch.argmax(y_truth[:, :, events], dim=2).flatten()
                # print(y_truth_label.size(), y_pred[:,:,events].squeeze(0).size())
                loss_mle = F.cross_entropy(y_pred[:, :, events].view((-1, y_pred[:, :, events].size(2))),
                                           torch.argmax(y_truth[:, :, events], dim=2).flatten(), weight=weights_final,
                                           reduction='sum')
                loss_mle += F.mse_loss(y_pred[:, :, duration_time_loc], y_truth[:, :, duration_time_loc],
                                       reduction='sum').tolist()
                loss_mle.backward()
                optimizerG.step()

                gen_loss_temp.append(loss_mle.tolist())
                # -----------------

        gen_loss_tot.append(np.mean(gen_loss_temp))
        # disc_loss_tot.append(np.mean(disc_loss_temp))
        gen_loss_temp = []
        # disc_loss_temp=[]
        if (i % 1 == 0):
            print('\n')
            # print("loss on generating suffix:", np.average )
            print("Iteration:", i, "The avg of Gen loss is:", gen_loss_tot[-1])
            # print("Iteration:",i, "The avg of Disc loss is:", disc_loss_tot[-1])

        # Applying validation after several epoches
        # Early stopping (checking for 5 times)
        if (i % 5 == 0):
            # #-----------------------------
            # weights_final[0] = max(weights_final[0]-5,1)

            # #-----------------------------
            model.eval()
            # accuracy, timestamp_accuracy, gen_loss_pred_validation  = Model_eval_test(rnnG, 'validation', events, selected_columns, duration_time_loc)
            # accuracy_on_validation.append(accuracy)
            gen_loss_pred_validation, dl_loss_validation, mae_loss_validation = model_eval_test(model, data_obj,'validation')
            loss_on_validation.append(gen_loss_pred_validation)
            dl_on_validation.append(dl_loss_validation)

            model.train()
            if (gen_loss_pred_validation <= np.min(loss_on_validation)):
                print("Best model on validation (entropy) is saved")
                # print("The validation set accuracy is:",accuracy)
                # accuracy_best = accuracy


                torch.save(model.state_dict(), os.path.join(data_obj.output_dir,'rnnG(validation entropy).m'))
                # torch.save(rnnG, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnG(validation)Timestamp.m")
                # torch.save(rnnD, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnD(validation)Timsestamp.m")

            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            if (dl_loss_validation >= np.max(dl_on_validation)):
                print("Best model on validation (DL) is saved")
                # print("The validation set accuracy is:",accuracy)
                # accuracy_best = accuracy

                torch.save(model.state_dict(), os.path.join(data_obj.output_dir,'rnnG(validation dl).m'))
                # torch.save(rnnG, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnG(validation)Timestamp.m")
                # torch.save(rnnD, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnD(validation)Timsestamp.m")

            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            if (len(loss_on_validation) > 30):
                if np.all(np.array(loss_on_validation[-29:]) > loss_on_validation[-30]):
                    print("Early stopping has halt the traning!!")
                    break

#############################################################################################
# Traning seq to seq model using GAN
def train_gan(model,rnnD, optimizerG,optimizerD, data_obj):


    unique_event = data_obj.unique_event
    train_suffix_loader_partition_list = data_obj.train_suffix_loader_partition_list
    selected_columns = data_obj.selected_columns
    duration_time_loc = data_obj.duration_time_loc
    weights_final = data_obj.weights_final


    model.train()
    epoch = 500
    events = list(np.arange(0, len(unique_event) + 1))
    # evetns = list(np.arange(1,len(unique_event)+1))
    gen_loss_pred = []
    disc_loss_tot = []
    gen_loss_tot = []
    gen_loss_temp = []
    disc_loss_temp = []
    accuracy_best = 0
    accuracy_on_validation = []
    loss_on_validation = []
    dl_on_validation = []
    mae_on_validation = []
    duration_time_loss = []

    model.train()
    for i in tqdm(range(epoch)):
        # train_suffix_loader = DataLoader(dataset=train_suffix_data, batch_size=batch, shuffle=True)
        for tl in train_suffix_loader_partition_list:
            train_suffix_loader = tl
            # MAX_ITER = tl.dataset[0][1].size()[0]
            # MAX_ITER = tl.dataset.tensors[1].size()[1]
            for mini_batch in iter(train_suffix_loader):
                # optimizerD.zero_grad()
                # optimizerG.zero_grad()

                x = mini_batch[0];
                y_truth = mini_batch[1];
                case_id = mini_batch[2]



                x_clone = x.clone()
                y_truth_clone = y_truth.clone()
                ###############################################################
                ###############################################################
                #with torch.autograd.set_detect_anomaly(True):
                #####GAN with GUMBLE SOFT
                optimizerD.zero_grad()
                x[:,:,events] = one_hot_to_gumble_soft(x[:,:,events])
                y_truth[:,:,events] = one_hot_to_gumble_soft(y_truth[:,:,events])

                #y_pred = model(x[:, :, selected_columns], y_truth[:, :, selected_columns], teacher_forcing_ratio=0.5)
                y_pred = model(src=x[:, :, selected_columns], trg=y_truth[:, :, selected_columns],teacher_forcing_ratio=0.1)

                t = np.power(.9, i)
                y_pred[:, :, events] = torch.nn.functional.gumbel_softmax(y_pred[:,:,events].detach(), tau=t)
                suffix_fake = y_pred
                suffix_real = y_truth[:, :, selected_columns]

                discriminator_synthetic_pred = rnnD(suffix_fake)
                discriminator_realistic_pred = rnnD(suffix_real)
                pr = discriminator_realistic_pred
                pf = discriminator_synthetic_pred


                # ll1 = -(1.0/pr.size()[0])*torch.sum(F.logsigmoid(pr))
                # ll2 = -(1.0/pf.size()[0])*torch.sum(F.logsigmoid(1.0 - pf))
                ll1 = -torch.mean(F.logsigmoid(pr))
                ll2 = -torch.mean(F.logsigmoid(1.0 - pf))
                discriminator_loss_tot = ll1 + ll2
                discriminator_loss_tot.backward(retain_graph=True)
                disc_loss_temp.append(discriminator_loss_tot.tolist())

                """ WGAN """
                # During discriminator forward-backward-update
                # discriminator_loss_tot = -(torch.mean(pr) - torch.mean(pf))
                # discriminator_loss_tot.backward(retain_graph=True)

                torch.nn.utils.clip_grad_norm_(rnnD.parameters(), max_norm=1)
                optimizerD.step()

                # Traning the generator
                optimizerG.zero_grad()

                # Generator
                pf = rnnD(suffix_fake)
                # gl = -(1.0/pf.size()[0])*torch.sum(F.logsigmoid(pf) - F.logsigmoid(1 - pf))
                gl = -torch.mean(F.logsigmoid(pf) - F.logsigmoid(1 - pf))
                # gl = -torch.mean(pf)
                gl.backward(retain_graph=True)
                gen_loss_temp.append(gl.tolist())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                # optimizerG.step()



                # ------------------
                # normal traning (MLE)
                # optimizerG.zero_grad()
                y_truth = y_truth_clone;
                x = x_clone
                y_pred = model(src=x[:, :, selected_columns], trg=y_truth[:, :, selected_columns],teacher_forcing_ratio=0.1)
                y_truth_label = torch.argmax(y_truth[:, :, events], dim=2).flatten()
                # print(y_truth_label.size(), y_pred[:,:,events].squeeze(0).size())
                loss_mle = F.cross_entropy(y_pred[:, :, events].view((-1, y_pred[:, :, events].size(2))),
                                           torch.argmax(y_truth[:, :, events], dim=2).flatten(), weight=weights_final)
                duration_time_loss.append(
                    F.mse_loss(y_pred[:, :, duration_time_loc], y_truth[:, :, duration_time_loc], reduction='sum').tolist())
                loss_mle.backward(retain_graph=True)
                optimizerG.step()

                # gen_loss_temp.append(loss_mle.tolist())
                # -----------------

        # """ Vanilla GAN """
        # # During discriminator forward-backward-update
        # D_loss = torch.mean(torch.log(D_real) - torch.log(1- D_fake))
        # # During generator forward-backward-update
        # G_loss = -torch.mean(torch.log(D_fake))

        # """ WGAN """
        # # During discriminator forward-backward-update
        # D_loss = -(torch.mean(D_real) - torch.mean(D_fake))
        # # During generator forward-backward-update
        # G_loss = -torch.mean(D_fake)

        gen_loss_tot.append(np.mean(gen_loss_temp))
        disc_loss_tot.append(np.mean(disc_loss_temp))
        gen_loss_temp = []
        disc_loss_temp = []
        if (i % 1 == 0):
            print('\n')
            # print("loss on generating suffix:", np.average )
            print("Iteration:", i, "The avg of Gen loss is:", gen_loss_tot[-1])
            print("cycle time loss:", np.mean(duration_time_loss))
            # print("Iteration:",i, "The avg of Disc loss is:", disc_loss_tot[-1])

        #Applying validation after several epoches
        # Early stopping (checking for 5 times)
        if (i % 5 == 0):
            model.eval()
            # accuracy, timestamp_accuracy, gen_loss_pred_validation  = Model_eval_test(rnnG, 'validation', events, selected_columns, duration_time_loc)
            # accuracy_on_validation.append(accuracy)
            gen_loss_pred_validation, dl_loss_validation, mae_loss_validation = model_eval_test(model, data_obj, 'validation')
            loss_on_validation.append(gen_loss_pred_validation)
            dl_on_validation.append(dl_loss_validation)
            mae_on_validation.append(dl_loss_validation / mae_loss_validation)

            # model.train()
            if (gen_loss_pred_validation <= np.min(loss_on_validation)):
                print("Best model on validation (entropy) is saved")
                # print("The validation set accuracy is:",accuracy)
                # accuracy_best = accuracy


                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation entropy gan).m'))
                # torch.save(rnnG, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnG(validation)Timestamp.m")
                # torch.save(rnnD, "C:/Users/ftaymouri/Desktop/Result/GAN/BPI2012/rnnD(validation)Timsestamp.m")

            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            if (dl_loss_validation >= np.max(dl_on_validation)):
                print("Best model on validation (DL) is saved")
                #torch.save(model.state_dict(), '/content/drive/My Drive/Deep Learing project/rnnG(validation dl gan).m')
                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation dl gan).m'))

            if (dl_loss_validation / mae_loss_validation >= np.max(mae_on_validation)):
                print("Best model on validation (DL/MAE) is saved")
                #torch.save(model.state_dict(), '/content/drive/My Drive/Deep Learing project/rnnG(validation mae gan).m')
                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation mae gan).m'))

            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            model.train()

            # if(len(loss_on_validation)>50):
            #   if np.all(np.array(loss_on_validation[-49:]) < loss_on_validation[-50]):
            #     print("Early stopping has halt the traning!!")



############################################################################################
def one_hot_to_gumble_soft(m):
  '''
  m: a 3 dimensional tensor,e.g., torch.Size([5, 4, 11])
  '''
  #print(torch.argmax(m, dim=2), torch.argmax(m, dim=2).view((m.size()[0], m.size()[1],-1)) )
  m[m==1] =.9
  m[m==0] = 0.1/(m.size()[2]-1)
  m= nn.functional.gumbel_softmax(m,dim=2, tau=0.001)
  return m

##############################################################################################

# Evaluating (New Game Encoder Decoder)
def model_eval_test(modelG, data_obj, mode):
    # set the evaluation mode (this mode is necessary if you train with batch, since in test the size of batch is different)
    rnnG = modelG
    # rnnD  = modelD
    rnnG.eval()
    # rnnD.eval()

    valid_suffix_loader_partition_list = data_obj.valid_suffix_loader_partition_list
    test_suffix_loader_partition_list = data_obj.test_suffix_loader_partition_list
    weights_final = data_obj.weights_final
    selected_columns = data_obj.selected_columns
    events = data_obj.events
    duration_time_loc = data_obj.duration_time_loc

    if (mode == 'validation'):
        # data_loader = validation_suffix_loader
        data_loader_partition_list = valid_suffix_loader_partition_list
    elif (mode == "test"):
        data_loader_partition_list = test_suffix_loader_partition_list
        # data_loader_partition_list = train_suffix_loader_partition_list
    elif (mode == 'test-validation'):
        data_loader_partition_list = test_suffix_loader_partition_list + valid_suffix_loader_partition_list

    predicted = []
    accuracy_record = []
    accuracy_time_stamp = []
    accuracy_time_stamp_per_event = {}
    accuracy_pred_per_event = {}
    mistakes = {}

    accuracy_record_2most_probable = []
    gen_loss_pred_validation = []

    d = []
    time_mae = []

    for dl in data_loader_partition_list:
        data_loader = dl
        # MAX_ITER = dl.dataset[0][1].size()[0]
        # MAX_ITER = dl.dataset.tensors[1].size()[1]
        for mini_batch in iter(data_loader):

            x = mini_batch[0];
            y_truth = mini_batch[1];
            case_id = mini_batch[2]

            # When we create mini batches, the length of the last one probably is less than the batch size, and it makes problem for the LSTM, therefore we skip it.
            # if(x.size()[0]<batch):
            #   continue
            # print("x.size()", x.size())

            # #Separating event and timestamp
            # y_truth_timestamp = y_truth[:,:,0].view(batch,1,-1)
            # y_truth_event = y_truth[:,:,1].view(batch,1,-1)

            # #Executing LSTM
            # y_pred = rnnG(x[:,:, selected_columns])
            # #print("y_pred:\n", y_pred, y_pred.size())

            if (mode == 'validation' or mode == 'test'):
                teacher_forcing_ratio = 0
            else:
                teacher_forcing_ratio = 0.5

            y_pred = rnnG(x[:, :, selected_columns], y_truth[:, :, selected_columns], teacher_forcing_ratio)

            # normal traning
            y_truth_label = torch.argmax(y_truth[:, :, events], dim=2).flatten()
            # print(y_truth_label.size(), y_pred[:,:,events].squeeze(0).size())
            loss_mle = F.cross_entropy(y_pred[:, :, events].view((-1, y_pred[:, :, events].size(2))),
                                       torch.argmax(y_truth[:, :, events], dim=2).flatten(), weight=weights_final)
            gen_loss_pred_validation.append(loss_mle.tolist())

            # computing DM distance on validation
            suffix_fake = y_pred
            suffix_real = y_truth

            r = torch.argmax(suffix_real[:, :, events], dim=2)
            f = torch.argmax(suffix_fake[:, :, events], dim=2)
            prefix = torch.argmax(x[:, :, events], dim=2)

            damerau = Damerau()
            # print("Prefix, Target, Predicted")
            k = 0
            for t, u, v in zip(prefix, r, f):
                k += 1

                if 0 in v.tolist():
                    u = u.tolist()[:u.tolist().index(0) + 1]
                    # u = u.tolist()
                    v = v.tolist()[:v.tolist().index(0) + 1]
                    # v = v.tolist()
                    t = t.tolist()
                else:
                    u = u.tolist()[:u.tolist().index(0) + 1]
                    # u = u.tolist()
                    v = v.tolist()
                    t = t.tolist()

                d.append(1.0 - float(damerau.distance(u, v)) / max(len(u), len(v)))

                # if (mode == 'test' or mode == ''):
                #
                #     print(t, u, v)

            time_mae.append(torch.mean(
                torch.abs(suffix_real[:, :, duration_time_loc] - suffix_fake[:, :, duration_time_loc])).tolist())

        # print("The DLV distance aveg for prefix of lenght?:", np.mean(d[-k:]))
        # print("The DLV distance aveg for prefix of lenght?:", np.mean(d[-x.size()[0]:]))

        # #-------------------------------------------------------------------------------

    rnnG.train()
    # rnnD.train()

    # if(mode == 'test'):
    #   pprint.pprint(mistakes)

    print("\n The corss entropy loss validation is:", np.mean(gen_loss_pred_validation))
    # print("Discriminator loss on validation:", np.mean(disc_loss_temp))
    print("The DLV distance aveg:", np.mean(d))
    print("MAE on validation:", np.mean(time_mae))
    # print("--------------------------------------------------------------")
    # print("Turth: first prediction, second prediction\n")
    # print("total number of predictions:", len(accuracy_record), np.sum(accuracy_record))
    # print("The accuracy of the model with the most probable event:", np.mean(accuracy_record))
    # print("The accuracy of the model with the 2 most probable events:", np.mean(accuracy_record_2most_probable))

    # print("The accuracy of prediction per event:")
    # print("Event, accuracy, frequency")
    # for k in accuracy_pred_per_event.keys():
    #   accuracy_pred_per_event[k] = [np.mean(accuracy_pred_per_event[k]), len(accuracy_pred_per_event[k])]
    # pprint.pprint(accuracy_pred_per_event )

    # print("The MAE(day) of timestamp prediction is:", np.mean(accuracy_time_stamp))
    # print("The MAE(day) of timestamp prediction per event:")
    # print("Event, accuracy, frequency")
    # for k in accuracy_time_stamp_per_event.keys():
    #   accuracy_time_stamp_per_event[k] = [np.mean(accuracy_time_stamp_per_event[k]), len(accuracy_time_stamp_per_event[k])]
    # pprint.pprint(accuracy_time_stamp_per_event )

    # return np.mean(accuracy_record), accuracy_time_stamp_per_event,  np.mean(gen_loss_pred_validation)
    # return np.mean(d)
    return np.mean(gen_loss_pred_validation), np.mean(d), np.mean(time_mae)

    # print("The MAE value is:", np.mean(accuracy_time_stamp))
    # return np.mean(accuracy_time_stamp)


