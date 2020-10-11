from __future__ import print_function, division
import os
import sys
import torch
from tqdm import tqdm
import numpy as np
np.set_printoptions(linewidth=1000)
import torch.nn as nn
import torch.nn.functional as F
import random
from similarity.damerau import Damerau
import preparation as pr
# device=torch.device('cuda:0')
# plt.style.use('ggplot')


class Encoder(nn.Module, pr.Preprocessing):
    def __init__(self, input_size, batch, hidden_size, num_layers, num_directions):
        nn.Module.__init__(self)


        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True, bidirectional=False)
        self.hid_dim = hidden_size
        self.n_layers = num_layers


    def forward(self, x):
        self.lstm.flatten_parameters()  # For running DataParallel
        output, (h, c) = self.lstm(x)

        return h, c

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

        begin_symbol = torch.ones((trg.size()[0], 1, trg.size()[2])).cuda()
        inp = begin_symbol

        # Iterating over the length of suffix
        for i in range(trg.size()[1]):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            # print(output.size())

            # decide if we are going to use teacher forcing or not
            # teacher_forcing_ratio =0.1
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force:
                inp = trg[:, i, :].view((trg.size()[0], 1, trg.size()[2]))
            else:
                inp = output

            begin_symbol = torch.cat((begin_symbol, output), dim=1)
        prediction = begin_symbol[:, 1:, :]

        return prediction



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
    unique_event = data_obj.unique_event
    train_suffix_loader_partition_list = data_obj.train_suffix_loader_partition_list
    selected_columns = data_obj.selected_columns
    duration_time_loc = data_obj.duration_time_loc
    weights_final = data_obj.weights_final



    model.train()
    epoch = 500
    events = list(np.arange(0, len(unique_event) + 1))
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
        for tl in train_suffix_loader_partition_list:
            train_suffix_loader = tl
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
        gen_loss_temp = []
        if (i % 1 == 0):
            print('\n')
            print("Iteration:", i, "The avg of Gen loss is:", gen_loss_tot[-1])

        # Applying validation after several epoches
        # Early stopping (checking for 30 times)
        if (i % 5 == 0):

            model.eval()
            gen_loss_pred_validation, dl_loss_validation, mae_loss_validation = model_eval_test(model, data_obj,'validation')
            loss_on_validation.append(gen_loss_pred_validation)
            dl_on_validation.append(dl_loss_validation)

            model.train()
            if (gen_loss_pred_validation <= np.min(loss_on_validation)):
                print("Best model on validation (entropy) is saved")

                torch.save(model.state_dict(), os.path.join(data_obj.output_dir,'rnnG(validation entropy).m'))


            # Checking whether the accuracy on validation is dropped or no (we consider the last 30 epoch)
            if (dl_loss_validation >= np.max(dl_on_validation)):
                print("Best model on validation (DL) is saved")

                torch.save(model.state_dict(), os.path.join(data_obj.output_dir,'rnnG(validation dl).m'))

            # Checking whether the accuracy on validation is dropped or no (we consider the last 30 epoch)
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
        for tl in train_suffix_loader_partition_list:
            train_suffix_loader = tl
            for mini_batch in iter(train_suffix_loader):


                x = mini_batch[0];
                y_truth = mini_batch[1];
                case_id = mini_batch[2]



                x_clone = x.clone()
                y_truth_clone = y_truth.clone()
                ###############################################################
                ###############################################################
                #####GAN with GUMBEL SOFT
                optimizerD.zero_grad()
                x[:,:,events] = one_hot_to_gumble_soft(x[:,:,events])
                y_truth[:,:,events] = one_hot_to_gumble_soft(y_truth[:,:,events])

                y_pred = model(src=x[:, :, selected_columns], trg=y_truth[:, :, selected_columns],teacher_forcing_ratio=0.1)

                t = np.power(.9, i)
                y_pred[:, :, events] = torch.nn.functional.gumbel_softmax(y_pred[:,:,events].detach(), tau=t)
                suffix_fake = y_pred
                suffix_real = y_truth[:, :, selected_columns]

                discriminator_synthetic_pred = rnnD(suffix_fake)
                discriminator_realistic_pred = rnnD(suffix_real)
                pr = discriminator_realistic_pred
                pf = discriminator_synthetic_pred



                ll1 = -torch.mean(F.logsigmoid(pr))
                ll2 = -torch.mean(F.logsigmoid(1.0 - pf))
                discriminator_loss_tot = ll1 + ll2
                discriminator_loss_tot.backward(retain_graph=True)
                disc_loss_temp.append(discriminator_loss_tot.tolist())


                torch.nn.utils.clip_grad_norm_(rnnD.parameters(), max_norm=1)
                optimizerD.step()

                # Traning the generator
                optimizerG.zero_grad()

                # Generator
                pf = rnnD(suffix_fake)
                gl = -torch.mean(F.logsigmoid(pf) - F.logsigmoid(1 - pf))
                gl.backward(retain_graph=True)
                gen_loss_temp.append(gl.tolist())

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)



                # ------------------
                # normal traning (MLE)
                y_truth = y_truth_clone;
                x = x_clone
                y_pred = model(src=x[:, :, selected_columns], trg=y_truth[:, :, selected_columns],teacher_forcing_ratio=0.1)
                y_truth_label = torch.argmax(y_truth[:, :, events], dim=2).flatten()
                loss_mle = F.cross_entropy(y_pred[:, :, events].view((-1, y_pred[:, :, events].size(2))),
                                           torch.argmax(y_truth[:, :, events], dim=2).flatten(), weight=weights_final)
                duration_time_loss.append(
                    F.mse_loss(y_pred[:, :, duration_time_loc], y_truth[:, :, duration_time_loc], reduction='sum').tolist())
                loss_mle.backward(retain_graph=True)
                optimizerG.step()
                # -----------------


        gen_loss_tot.append(np.mean(gen_loss_temp))
        disc_loss_tot.append(np.mean(disc_loss_temp))
        gen_loss_temp = []
        disc_loss_temp = []
        if (i % 1 == 0):
            print('\n')
            print("Iteration:", i, "The avg of Gen loss is:", gen_loss_tot[-1])
            print("cycle time loss:", np.mean(duration_time_loss))


        #Applying validation after several epoches
        # Early stopping (checking for 30 times)
        if (i % 5 == 0):
            model.eval()
            gen_loss_pred_validation, dl_loss_validation, mae_loss_validation = model_eval_test(model, data_obj, 'validation')
            loss_on_validation.append(gen_loss_pred_validation)
            dl_on_validation.append(dl_loss_validation)
            mae_on_validation.append(dl_loss_validation / mae_loss_validation)

            # model.train()
            if (gen_loss_pred_validation <= np.min(loss_on_validation)):
                print("Best model on validation (entropy) is saved")



                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation entropy gan).m'))


            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            if (dl_loss_validation >= np.max(dl_on_validation)):
                print("Best model on validation (DL) is saved")
                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation dl gan).m'))

            if (dl_loss_validation / mae_loss_validation >= np.max(mae_on_validation)):
                print("Best model on validation (DL/MAE) is saved")
 
                torch.save(model.state_dict(), os.path.join(data_obj.output_dir, 'rnnG(validation mae gan).m'))

            # Checking whether the accuracy on validation is dropped or no (we consider the last 5 epoch)
            model.train()



############################################################################################
def one_hot_to_gumble_soft(m):
  '''
  m: a 3 dimensional tensor,e.g., torch.Size([5, 4, 11])
  '''
  m[m==1] =.9
  m[m==0] = 0.1/(m.size()[2]-1)
  m= nn.functional.gumbel_softmax(m,dim=2, tau=0.001)
  return m

##############################################################################################

# Evaluating (New Game Encoder Decoder)
def model_eval_test(modelG, data_obj, mode):
    # set the evaluation mode (this mode is necessary if you train with batch, since in test the size of batch is different)
    rnnG = modelG
    rnnG.eval()


    valid_suffix_loader_partition_list = data_obj.valid_suffix_loader_partition_list
    test_suffix_loader_partition_list = data_obj.test_suffix_loader_partition_list
    weights_final = data_obj.weights_final
    selected_columns = data_obj.selected_columns
    events = data_obj.events
    duration_time_loc = data_obj.duration_time_loc

    if (mode == 'validation'):
        data_loader_partition_list = valid_suffix_loader_partition_list
    elif (mode == "test"):
        data_loader_partition_list = test_suffix_loader_partition_list
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
        for mini_batch in iter(data_loader):

            x = mini_batch[0];
            y_truth = mini_batch[1];
            case_id = mini_batch[2]


            if (mode == 'validation' or mode == 'test'):
                teacher_forcing_ratio = 0
            else:
                teacher_forcing_ratio = 0.5

            y_pred = rnnG(x[:, :, selected_columns], y_truth[:, :, selected_columns], teacher_forcing_ratio)

            # normal traning
            y_truth_label = torch.argmax(y_truth[:, :, events], dim=2).flatten()
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

            time_mae.append(torch.mean(
                torch.abs(suffix_real[:, :, duration_time_loc] - suffix_fake[:, :, duration_time_loc])).tolist())


        # #-------------------------------------------------------------------------------

    rnnG.train()


    print("\n The corss entropy loss validation is:", np.mean(gen_loss_pred_validation))
    print("The DLV distance aveg:", np.mean(d))
    print("MAE on validation:", np.mean(time_mae))


    return np.mean(gen_loss_pred_validation), np.mean(d), np.mean(time_mae)



