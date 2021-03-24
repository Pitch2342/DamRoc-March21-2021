# Title : MainScripts.py
# Description 
    # Contains the main code and neural network
    # All helper functions are imported
    # A fully connected Deep Neural Network with 3 layers (specific layer size can be modified)
    # Is called from run_expts.py to execute for every different required combination of attack and defense
    # Gets data from a folder 'helper files' in same path and saves results to 'result_files' folder
# Author : Pitch2342
# Date : March 15, 2021

# Imports
import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.utils import load_parameters, stack_tensors
from datasets.datasets import load_data
from inner_maximizers.inner_maximizers import inner_maximizer
from nets.ff_classifier import build_ff_classifier
from coverage.covering_number import Coverage
import losswise
import time
import json
import numpy as np

# Loading parameters
parameters = load_parameters("parameters.ini")
is_cuda = eval(parameters["general"]["is_cuda"])
if is_cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = parameters["general"]["gpu_device"]

assertion_message = "Set this flag off to train models."
assert eval(parameters['dataset']['generate_feature_vector_files']) is False, assertion_message

log_interval = int(parameters["general"]["log_interval"])
num_epochs = int(parameters["hyperparam"]["ff_num_epochs"])
# is_losswise = eval(parameters["general"]["is_losswise"])
# is_synthetic_dataset = eval(parameters["general"]["is_synthetic_dataset"])

training_method = parameters["general"]["training_method"]
evasion_method = parameters["general"]["evasion_method"]
experiment_suffix = parameters["general"]["experiment_suffix"]
experiment_name = "[training:%s|evasion:%s]_%s" % (training_method, evasion_method,
                                                   experiment_suffix)

print("Training Method:%s, Evasion Method:%s" % (training_method, evasion_method))

seed_val = int(parameters["general"]["seed"])

random.seed(seed_val)
torch.manual_seed(seed_val)
np.random.seed(seed_val)

evasion_iterations = int(parameters['hyperparam']['evasion_iterations'])

save_every_epoch = eval(parameters['general']['save_every_epoch'])

train_model_from_scratch = eval(parameters['general']['train_model_from_scratch'])
load_model_weights = eval(parameters['general']['load_model_weights'])
model_weights_path = parameters['general']['model_weights_path']

# Data Loading
train_dataloader_dict, valid_dataloader_dict, test_dataloader_dict, num_features = load_data(
    parameters)

# Coverage Definitions
num_samples = len(train_dataloader_dict["malicious"].dataset)
bscn = Coverage(num_samples)
gscn = Coverage(num_samples)

# Classifier Build
if load_model_weights:
    print("Loading Model Weights From: {path}".format(path=model_weights_path))
    model = torch.load(model_weights_path)

else:
    model = build_ff_classifier(
        input_size=num_features,
        hidden_1_size=int(parameters["hyperparam"]["ff_h1"]),
        hidden_2_size=int(parameters["hyperparam"]["ff_h2"]),
        hidden_3_size=int(parameters["hyperparam"]["ff_h3"]))
    print(model)

if is_cuda:
    torch.cuda.manual_seed(int(parameters["general"]["seed"]))
    model = model.cuda()

# Define Loss functions and optimizers
loss_fct = nn.NLLLoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=float(parameters["hyperparam"]["ff_learning_rate"]))

# TRAINING FUNCTION
# To train using a given attack method <mentioned in parameters.ini>
def train(epoch):
    model.train()
    total_correct = 0.
    total_loss = 0.
    total = 0.

    current_time = time.time()

    # if is_synthetic_dataset:
    #     # since generation of synthetic data set is random, we'd like them to be the same over epochs
    #     torch.manual_seed(seed_val)
    #     random.seed(seed_val)
    print("dataloaderr",train_dataloader_dict["benign"].dataset, train_dataloader_dict["malicious"].dataset)
    for batch_idx, ((bon_x, bon_y), (mal_x, mal_y)) in enumerate(
            zip(train_dataloader_dict["benign"], train_dataloader_dict["malicious"])):
        # Check for adversarial learning
        print("This is bacth_idx",batch_idx)
        print(bon_x)
        print(bon_y)
        print(mal_x)
        print(mal_y)
        mal_x1 = inner_maximizer(
            mal_x, mal_y, model, loss_fct, iterations=evasion_iterations, method=training_method)
        print("THis is mal_X",mal_x1)
        # stack input
        if is_cuda:
            x = Variable(stack_tensors(bon_x, mal_x, mal_x1).cuda())
            y = Variable(stack_tensors(bon_y, mal_y, mal_y).cuda())
        else:
            x = Variable(stack_tensors(bon_x, mal_x, mal_x1))
            y = Variable(stack_tensors(bon_y, mal_y, mal_y))
        print("This is Train Forward Pass",x, y, model )
        
        # forward pass
        print("FORWARD X")
        print(x)
        y_model = model(x)

        # backward pass
        optimizer.zero_grad()
        loss = loss_fct(y_model, y).mean()
        loss.backward()
        optimizer.step()

        # predict pass
        _, predicted = torch.topk(y_model, k=1)
        correct = predicted.data.eq(y.data.view_as(predicted.data)).cpu().sum()

        # metrics
        print("LOSS LINE 148")
        print(len(y))
        print(y)
        print(total_loss)
        print(loss.data)
        print(loss.data.item())
        # total_loss += loss.data[0] * len(y)
        total_loss += loss.data.item() * len(y)

        total_correct += correct
        total += len(y)

#         bscn.update_numerator_batch(batch_idx, mal_x)
        print("COVERING UPDATES",mal_x.size(0))
        for i in range(mal_x.size(0)):
            print("UPDATING NORMAL MALS")
            a = bscn.update(mal_x[i])
            print("UPDATINGS ADVS")
            b = bscn.update(mal_x1[i])
            print("UPDATING GOOD SPOTS")
            c = gscn.update(bon_x[i])
            print(a,b,c)

        if batch_idx % log_interval == 0:

            print("Time Taken:", time.time() - current_time)
            current_time = time.time()

            print(
                "Train Epoch ({}) | Batch ({}) | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}\tBatch Accuracy: {:.1f}%\t BSCN: {:.12f}".
                # format(epoch, batch_idx, batch_idx * len(x),
                #        len(train_dataloader_dict["malicious"].dataset) +
                #        len(train_dataloader_dict["benign"].dataset),
                #        100. * batch_idx / len(train_dataloader_dict["benign"]), loss.data[0],
                #        100. * correct / len(y), bscn.ratio()))
                format(epoch, batch_idx, batch_idx * len(x),
                       len(train_dataloader_dict["malicious"].dataset) +
                       len(train_dataloader_dict["benign"].dataset),
                       100. * batch_idx / len(train_dataloader_dict["benign"]), loss.data.item(),
                       100. * correct / len(y), bscn.Covering_value()))

    # if is_losswise:
    #     graph_accuracy.append(epoch, {
    #         "train_accuracy_%s" % experiment_name: 100. * total_correct / total
    #     })
    #     graph_loss.append(epoch, {"train_loss_%s" % experiment_name: total_loss / total})
    #     graph_coverage.append(epoch, {"train_coverage_%s" % experiment_name: bscn.ratio()})

    model_filename = "{name}_epoch_{e}".format(name=experiment_name, e=epoch)

    # if save_every_epoch:
    #     torch.save(model, os.path.join("model_weights", model_filename))

def check_one_category(category="benign", is_validate=False, is_evade=False,
                       evade_method='dfgsm_k'):
    """
    test the model in terms of loss and accuracy on category, this function also allows to perform perturbation
    with respect to loss to evade
    :param category: benign or malicious dataset
    :param is_validate: validation or testing dataset
    :param is_evade: to perform evasion or not
    :param evade_method: evasion method (we can use on of the inner maximier methods), it is only relevant if is_evade
      is True
    :return:
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    evasion_mode = ""

    # if is_synthetic_dataset:
    #     # since generation of synthetic data set is random, we'd like them to be the same over epochs
    #     torch.manual_seed(seed_val)
    #     random.seed(seed_val)

    if is_validate:
        dataloader = valid_dataloader_dict[category]
    else:
        dataloader = test_dataloader_dict[category]

    for batch_idx, (x, y) in enumerate(dataloader):
        #
        if is_evade:
            x = inner_maximizer(
                x, y, model, loss_fct, iterations=evasion_iterations, method=evade_method)
            evasion_mode = "(evasion using %s)" % evade_method
        # stack input
        if is_cuda:
            x = Variable(x.cuda())
            y = Variable(y.cuda())
        else:
            x = Variable(x)
            y = Variable(y)
        print("Check one CategoryForward Pass", x, y, model)
        # forward pass
        y_model = model(x)

        # loss pass
        loss = loss_fct(y_model, y).mean()

        # predict pass
        _, predicted = torch.topk(y_model, k=1)
        correct = predicted.data.eq(y.data.view_as(predicted.data)).cpu().sum()

        # metrics
        # total_loss += loss.data[0] * len(y)
        total_loss += loss.data.item() * len(y)
        total_correct += correct
        total += len(y)

    print("{} set for {} {}: Average Loss: {:.4f}, Accuracy: {:.2f}%".format(
        "Valid" if is_validate else "Test", category, evasion_mode, total_loss / total,
        total_correct * 100. / total))

    return total_loss, total_correct, total

def test(epoch, is_validate=False):
    """
    Function to be used for both testing and validation
    :param epoch: current epoch
    :param is_validate: is the testing done on the validation dataset
    :return: average total loss, dictionary of the metrics for both bon and mal samples
    """
    # test for accuracy and loss
    bon_total_loss, bon_total_correct, bon_total = check_one_category(
        category="benign", is_evade=False, is_validate=is_validate)
    mal_total_loss, mal_total_correct, mal_total = check_one_category(
        category="malicious", is_evade=False, is_validate=is_validate)

    # test for evasion on malicious sample
    evade_mal_total_loss, evade_mal_total_correct, evade_mal_total = check_one_category(
        category="malicious", is_evade=True, evade_method=evasion_method, is_validate=is_validate)

    total_loss = bon_total_loss + mal_total_loss
    total_correct = bon_total_correct + mal_total_correct
    total = bon_total + mal_total

    dataset_type = "valid" if is_validate else "test"

    print("{} set overall: Average Loss: {:.4f}, Accuracy: {:.2f}%".format(
        dataset_type, total_loss / total, total_correct * 100. / total))

    # if is_losswise:
    #     graph_accuracy.append(
    #         epoch, {
    #             "%s_accuracy_%s" % (dataset_type, experiment_name): 100. * total_correct / total
    #         })
    #     graph_loss.append(epoch, {
    #         "%s_loss_%s" % (dataset_type, experiment_name): total_loss / total
    #     })
    #     graph_evasion.append(
    #         epoch, {
    #             "%s_evasion_%s" % (dataset_type, experiment_name):
    #             100 * (evade_mal_total - evade_mal_total_correct) / evade_mal_total
    #         })

    metrics = {
        "bscn_ratio": bscn.Covering_value(),
        "mal": {
            "total_loss": mal_total_loss,
            "total_correct": mal_total_correct,
            "total": mal_total,
            "evasion": {
                "total_loss": evade_mal_total_loss,
                "total_correct": evade_mal_total_correct,
                "total": evade_mal_total
            }
        },
        "bon": {
            "total_loss": bon_total_loss,
            "total_correct": bon_total_correct,
            "total_evade": None,
            "total": bon_total
        }
    }
    print(metrics)

    return (bon_total_loss + max(mal_total_loss, evade_mal_total_loss)) / total, metrics


if __name__ == "__main__":

    # if not os.path.exists("result_files"):
    #     os.mkdir("result_files")

    _metrics = None
    session = None
    if train_model_from_scratch:
        best_valid_loss = float("inf")
        for _epoch in range(num_epochs):
            # train
            train(_epoch)

            # validate
            valid_loss, _ = test(_epoch, is_validate=True)

            # keep the best parameters w.r.t validation and check the test set
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                _, _metrics = test(_epoch, is_validate=False)

                bscn_to_save = bscn.Covering_value()
                # with open(os.path.join("result_files", "%s_bscn_test.txt" % experiment_name), "w") as f:
                #     f.write(str(bscn_to_save))
                print("BSCN SAVING")
                print(bscn_to_save)
                print("MODEL SAVE {}".format(experiment_name))
                # torch.save(model, os.path.join("helper_files", "%s-model.pt" % experiment_name))
            elif _epoch % log_interval == 0:
                test(_epoch, is_validate=False)

    else:
        _, _metrics = test(0)

    # with open(os.path.join("result_files/" + experiment_name + ".json"), "w") as result_file:
    #     json.dump(_metrics, result_file)
    print("METRICS TO DUMP")
    print(_metrics)

    # if is_losswise:
    #     session.done()
