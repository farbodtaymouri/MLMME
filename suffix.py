import collections
import torch
import torch.nn.functional as F
import numpy as np
import xlsxwriter
import os
import copy
import pickle
from similarity.damerau import Damerau

#def suffix_generate(model, events, test_suffix_loader_partition_list, selected_columns, candidate_num=1):
def suffix_generate(model, data_obj, candidate_num=1):
    '''
    model: The generator of GANs
    events: A List of name of acitvity names including the end of trace, i.e., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    test_suffix_loader: A dataset contaning prefixes and suffixes in one hot encoding in a batch structure. For example: prefix_from_begin.size()= torch.Size([627, 4, 14])
    '''

    test_suffix_loader_partition_list = data_obj.test_suffix_loader_partition_list
    selected_columns = data_obj.selected_columns
    events = data_obj.events
    duration_time_loc = data_obj.duration_time_loc
    average_trace_length = data_obj.average_trace_length
    std_trace_length = data_obj.std_trace_length



    rnnG = model
    # candidate_num = len(events)
    # candidate_num = 5
    rnnG.eval()



    suffix_pred_list = []
    suffix_truth_list = []



    # Nested dictionary
    suffix_pred_dic = collections.defaultdict(dict)  # Its like {0: {(8, 6, 0): [[0], [8, 0], [8, 8, 0]]},....}
    suffix_pred_remain_time_dic = collections.defaultdict(dict)
    suffix_truth_remain_time_dic = collections.defaultdict(dict)
    suffix_prefix_dic = collections.defaultdict(dict)

    # # An index to keep track of suffixes
    # k=0
    for tl in test_suffix_loader_partition_list:
        test_suffix_loader = tl
        for mini_batch in iter(test_suffix_loader):
            # Prefix and suffix
            x = mini_batch[0];
            y_truth = mini_batch[1];
            case_id = mini_batch[2]


            # Start iterating over each prefix in the minibatch, until all for all prefixes the next acticity is 0 (end of trace)
            # Iterating for each prefix inside the minibatch

            for i in range(0, x[:, :, selected_columns].size()[0]):
                # An index to keep track of suffixes (an integer)
                k = case_id[i].tolist()
                # print(x[:,:, selected_columns])
                element = x[i, :, selected_columns]
                condition = 1
                input_x = element.view((1, element.size()[0], element.size()[1]))



                temp = torch.argmax(y_truth[i, :, events], dim=1).tolist()

                suffix_truth_list = temp[0:temp.index(0) + 1]
                # print(suffix_truth_list[0])
                # suffix_pred_dic_temp[tuple(suffix_truth_list)] = []

                pref = tuple(torch.argmax(x[i, :, events], dim=1).tolist())
                suffix_prefix_dic[k][tuple(suffix_truth_list)] = pref
                suffix_pred_dic[k][tuple(suffix_truth_list)] = [[] for j in range(0, candidate_num)]
                suffix_pred_remain_time_dic[k][tuple(suffix_truth_list)] = [[] for j in range(0, candidate_num)]
                suffix_truth_remain_time_dic[k][tuple(suffix_truth_list)] = [
                    np.sum(y_truth[i, :, duration_time_loc].tolist())]

                candidate_suffix = []
                # input_x[:,:,events] = one_hot_to_gumble_soft(input_x[:,:,events])

                if (hasattr(rnnG, 'module')):   #This is for when we use Dataparallel to run on several GPU
                    hidden, cell = rnnG.module.encoder(input_x)
                else:
                    hidden, cell = rnnG.encoder(input_x)
                # trg = trg[:,:,selected_columns]
                # print(trg.size(), begin_sentence.size())
                trg = y_truth[:, :, selected_columns]
                begin_symbol = torch.ones((1, 1, trg.size()[2])).cuda()
                input_x = begin_symbol
                while (condition):
                    if (hasattr(rnnG, 'module')):  # This is for when we use Dataparallel to run on several GPU
                        y_pred, hidden, cell = rnnG.module.decoder(input_x, hidden, cell)
                    else:
                        y_pred, hidden, cell = rnnG.decoder(input_x, hidden, cell)




                    candidate_suffix = beam2(candidate_suffix,
                                             y_pred[:, -1, :].view((y_pred.size()[0], -1, y_pred.size()[2])), events,
                                             size=candidate_num)

                    for j in range(candidate_num):
                        # In the first iteration the number of generated suffixes mught be less than the proposed number of candidates
                        if (len(candidate_suffix) < candidate_num):
                            suffix_pred_remain_time_dic[k][tuple(suffix_truth_list)][j].append(
                                np.abs(np.round(y_pred[:, y_pred.size()[1] - 1, duration_time_loc][-1].tolist(), 4)))
                        elif (len(candidate_suffix) == candidate_num and candidate_suffix[j][0][-1] != 0):
                            suffix_pred_remain_time_dic[k][tuple(suffix_truth_list)][j].append(
                                np.abs(np.round(y_pred[:, y_pred.size()[1] - 1, duration_time_loc][-1].tolist(), 4)))



                    count = 0
                    for j in range(len(candidate_suffix)):
                        estimated_suffix_len = int(average_trace_length / element.size()[0])
                        if (len(candidate_suffix[j][0]) > average_trace_length + (
                                estimated_suffix_len + 2) * std_trace_length):
                            candidate_suffix[j][0].append(0)

                        count += candidate_suffix[j][0][-1]

                    if (count == 0):
                        condition = 0
                        for j in range(0, candidate_num):
                            suffix_pred_dic[k][tuple(suffix_truth_list)][j] = candidate_suffix[j][0][
                                                                              :candidate_suffix[j][0].index(0) + 1]



                    ########################################
                    y_pred_next = y_pred[:, -1, :].view((y_pred.size()[0], -1, y_pred.size()[2])).clone()
                    y_pred_next[:, :, events] = F.one_hot(
                        torch.argmax(F.softmax(y_pred_next[:, :, events], dim=2), dim=2),
                        num_classes=len(events)).float()
                    input_x = y_pred_next

                    ########################################

                # k+=1
            # break
        # break

    #--------------------------
    out = os.path.join(data_obj.output_dir, 'suffix-generated '+str(candidate_num) + data_obj.dataset_name + data_obj.training_mode+ '.pkl')
    pickle.dump((suffix_pred_dic, suffix_pred_remain_time_dic, suffix_truth_remain_time_dic, suffix_prefix_dic), open(out, "wb"))
    print('Suffix Generation is done!')






    data_obj.suffix_pred_dic = suffix_pred_dic
    data_obj.suffix_pred_remain_time_dic = suffix_pred_remain_time_dic
    data_obj.suffix_truth_remain_time_dic = suffix_truth_remain_time_dic
    data_obj.suffix_prefix_dic = suffix_prefix_dic

    #return suffix_pred_dic, suffix_pred_remain_time_dic, suffix_truth_remain_time_dic, suffix_prefix_dic

#################################################################################################################
def beam2(candidate, y_pred, events, size=3):
    '''
    candidate: It can be an empty list for the first call, i.e., [], or a list of events with their scores for the next calles, e.g., [ [[2], 2.757077693939209], [[9], 2.7756776809692383],[[3], 3.1773548126220703] ]
    y_pred: The unnormalized output out of the neural net.
    events: A list containing the name of events, e.g., [0,1,2,3,4]
    size: The width of beam search

    Return: candidate
    '''
    if (len(candidate) == 0):
        temp = torch.sort(F.softmax(y_pred[:, y_pred.size()[1] - 1, events], dim=1), descending=True)[1][0][
               0:size].tolist()
        # temp = torch.sort(F.gumbel_softmax(y_pred[:,y_pred.size()[1]-1,events],dim=1, tau=0.001), descending= True)[1][0][0:size].tolist()
        candidate = [[[e], -F.log_softmax(y_pred[:, y_pred.size()[1] - 1, events], dim=1)[0][e].tolist()] for e in temp]
        # [[[0], 0.002113249042700036],
        # [[21], 6.457719656285117],
        # [[22], 7.519231708374177]]
    else:
        temp = []
        for i in range(len(candidate)):
            if (candidate[i][0][-1] == 0):
                continue
            for e in events:
                suffix = candidate[i][0] + [e]
                score = candidate[i][1] - F.log_softmax(y_pred[:, y_pred.size()[1] - 1, events], dim=1)[0][e].tolist()
                # candidate[i][1]+= -F.log_softmax(y_pred[:,y_pred.size()[1]-1,events],dim=1)[0][e].tolist()
                temp.append([suffix, score])

        # candidate = sorted(temp,  key = lambda x: x[1])[0:size]
        for c in candidate:
            if (c[0][-1] == 0):
                temp.append(c)
        candidate = sorted(temp, key=lambda x: x[1])[0:size]

    return candidate

#################################################################################################################

def suffix_similarity(data_obj, beam_size):
    '''
    suffix_pred_list: List of suffix predictions. Its a nested list like [[8, 6, 0], [6, 0], [6, 0]] where each nested list is related to a prefix
    suffix_truth_list: List of ground trtuth suffix. Its a nested list like [[8, 9, 8, 6, 0], [9, 8, 6, 0], [6, 0]] where each nested list is related to a prefix
    '''

    suffix_pred_dic = data_obj.suffix_pred_dic
    suffix_pred_remain_time_dic = data_obj.suffix_pred_remain_time_dic
    suffix_truth_remain_time_dic = data_obj.suffix_truth_remain_time_dic
    suffix_prefix_dic = data_obj.suffix_prefix_dic
    duration_time_max = data_obj.duration_time_max



    damerau = Damerau()
    distance_values = []
    timeError = []

    # Writing to excell file
    workbook = xlsxwriter.Workbook(os.path.join(data_obj.output_dir, data_obj.dataset_name+ data_obj.training_mode+'similarity'+str(beam_size)+'.xlsx'))



    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, "Case_ID")
    worksheet.write(0, 1, "Prefix")
    worksheet.write(0, 2, "Suffix Predicted")
    worksheet.write(0, 3, "Suffix ground Truth")
    worksheet.write(0, 4, "DLS measure")
    worksheet.write(0, 5, "MAE(days)")
    worksheet.write(0, 6, "Best Index solution")
    row = 1;
    col = 0

    for i in suffix_pred_dic.keys():

        for k in suffix_pred_dic[i].keys():
            d = []
            for j in range(len(suffix_pred_dic[i][k])):
            #for j in range(0, beam_size):
                pred = suffix_pred_dic[i][k][j]
                truth = k
                d.append(1.0 - float(damerau.distance(pred, truth)) / max(len(pred), len(truth)))
            distance_values.append(max(d))
            index_max = np.argmax(d)

            # timeError.append(timeDiff(suffix_pred_dic,k, index_max))
            timeError.append(
                np.abs(np.sum(suffix_pred_remain_time_dic[i][k][index_max]) - suffix_truth_remain_time_dic[i][k][0]))
            worksheet.write(row, col, str(i))
            worksheet.write(row, col + 1, str(suffix_prefix_dic[i][k]))
            worksheet.write(row, col + 2, str(suffix_pred_dic[i][k][index_max]))
            worksheet.write(row, col + 3, str(k))
            worksheet.write(row, col + 4, str(max(d)))
            worksheet.write(row, col + 5, str(timeError[-1]))
            worksheet.write(row, col + 6, str(index_max))
            row += 1

    worksheet.write(row + 5, col, "The average of DLS: " + str(np.mean(distance_values)))
    worksheet.write(row + 6, col, "The mediane of MAES: " + str(np.median(timeError)))
    worksheet.write(row + 7, col, "The average of MAE, normalized, of cycle time: " + str(np.mean(timeError)))
    worksheet.write(row + 8, col, "The average of MAE of cycle time: " + str(np.mean(timeError) * duration_time_max))



    workbook.close()



    return distance_values, timeError
