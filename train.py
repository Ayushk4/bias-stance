# Load Packages and setup wandb
from params import params
import wandb
if params.wandb:
    wandb.init(project="Biasstance", name=params.run)
    wandb.config.update(params)

from bertloader import StanceDataset
import json, os, random

import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW, AutoModel

from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

def train(model, dataset, criterion):
    model.train()
    train_losses = []
    num_batch = 0

    for batch in dataset:
        (texts, stances, att_masks, token_type) = batch
        preds = model(texts, att_masks, token_type)
        loss = criterion(preds, stances)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
      #  scheduler.step()

        if num_batch % 100 == 0:
            print("Train loss at {}:".format(num_batch), loss.item())

        num_batch += 1
        train_losses.append(loss.item())

    return np.average(train_losses)

def evaluate(model, dataset, criterion, target_names):
    model.eval()
    valid_losses = []
    predicts = []
    gnd_truths = []

    with torch.no_grad():
        for batch in dataset:
            (texts, stances, att_masks, token_type) = batch
            preds = model(texts, att_masks, token_type)

            loss = criterion(preds, stances)

            predicts.extend(torch.max(preds, axis=1)[1].tolist())
            gnd_truths.extend(stances.tolist())
            valid_losses.append(loss.item())

    assert len(predicts) == len(gnd_truths)

    confuse_mat = confusion_matrix(gnd_truths, predicts)
    if params.dummy_run:
        classify_report = {"hi": {"fake": 1.2}}
    else:
        classify_report = classification_report(gnd_truths, predicts, target_names=target_names, output_dict=True)

    mean_valid_loss = np.average(valid_losses)
    print("Valid_loss", mean_valid_loss)
    print(confuse_mat)

    for labl in target_names:
        print(labl,"F1-score:", classify_report[labl]["f1-score"])
    print("Accu:", classify_report["accuracy"])
    print("F1-Weighted", classify_report["weighted avg"]["f1-score"])
    print("F1-Avg", classify_report["macro avg"]["f1-score"])

    return mean_valid_loss, confuse_mat ,classify_report


########## Load dataset #############
dataset_object = StanceDataset()
train_dataset = dataset_object.train_dataset
eval_dataset = dataset_object.eval_dataset

if params.dummy_run:
    eval_dataset = train_dataset
    target_names = []
else:
    eval_dataset = dataset_object.eval_dataset
    if params.dataset_name == "enc":
        target_names = [dataset_object.id2stance[id_] for id_ in range(0, 2)]
    elif params.dataset_name in ["16se", "mt1", "mt2"]:
        target_names = [dataset_object.id2stance[id_] for id_ in range(0, 3)]
    else:
        target_names = [dataset_object.id2stance[id_] for id_ in range(0, 4)]


print("Dataset created")
os.system("nvidia-smi")


########## Create model #############

class BERTStance(nn.Module):
    def __init__(self, num_stances=4):
        super(BERTStance, self).__init__()
        self.bert = AutoModel.from_pretrained(params.bert_type)
        self.drop = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_stances)
    def forward(self, text, att_mask, token_type):
        _, pooled = self.bert(text, attention_mask=att_mask, token_type_ids=token_type)
        return self.classifier(self.drop(pooled))

if params.dataset_name == "enc":
    model = BERTStance(2)
elif params.dataset_name in ["16se", "mt1", "mt2"]:
    model = BERTStance(3)
else:
    model = BERTStance(4)
import os
print("Model created")
os.system("nvidia-smi")
embedding_size = model.bert.embeddings.word_embeddings.weight.size(1)
new_embeddings = torch.FloatTensor(3, embedding_size).uniform_(-0.1, 0.1)
new_embedding_weight = torch.cat((model.bert.embeddings.word_embeddings.weight.data,new_embeddings), 0)
model.bert.embeddings.word_embeddings.weight.data = new_embedding_weight
print("Embedding Shape:", model.bert.embeddings.word_embeddings.weight.data.size())

print(sum(p.numel() for p in model.parameters()))
model = model.to(params.device)
print("Detected", torch.cuda.device_count(), "GPUs!")
# model = torch.nn.DataParallel(model)

if params.wandb:
    wandb.watch(model)

########## Optimizer & Loss ###########

#criterion = torch.nn.CrossEntropyLoss(weight=dataset_object.criterion_weights, reduction='sum')
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)

# valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)

for epoch in range(params.n_epochs):
    print("\n\n========= Beginning", epoch+1, "epoch ==========")

    train_loss = train(model, train_dataset, criterion)
    if not params.dummy_run:
        print("EVALUATING:")
        valid_loss, confuse_mat, classify_report = evaluate(model, eval_dataset, criterion, target_names)
    else:
        valid_loss = 0.0

    if not params.dummy_run and params.wandb:
        wandb_dict = {}
        for labl in target_names:
            for metric, val in classify_report[labl].items():
                if metric != "support":
                    wandb_dict[labl + "_" + metric] = val

        wandb_dict["F1-Weighted"] = classify_report["weighted avg"]["f1-score"]
        wandb_dict["F1-Avg"] = classify_report["macro avg"]["f1-score"]

        wandb_dict["Accuracy"] = classify_report["accuracy"]

        wandb_dict["Train_loss"] = train_loss
        wandb_dict["Valid_loss"] = valid_loss

        wandb.log(wandb_dict)

    epoch_len = len(str(params.n_epochs))
    print_msg = (f'[{epoch:>{epoch_len}}/{params.n_epochs:>{epoch_len}}]     ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
    print(print_msg)

if params.test_mode:
    basepath = os.path.join("/".join(os.path.realpath(__file__).split('/')[:-1]),
                            "saves")
    folder_name = params.dataset_name
    if params.dataset_name == 'wtwt':
        folder_name += "_" + params.target_merger 
    if params.notarget:
        folder_name += "_" + "notarget"
    else:
        folder_name += "_" + "target"
    folder_name = os.path.join(basepath, folder_name)
    print(folder_name)
    if os.path.isdir(folder_name):
        os.system("rm -rf " + folder_name)
    os.mkdir(folder_name)

    # Store params
    json.dump(vars(params), open(os.path.join(folder_name, "params.json"), 'w+'))

    # Save model
    torch.save(model.state_dict(), os.path.join(folder_name, "model.pt"))

    # Store logs (accuracy)
    logs = {"Accu:": classify_report["accuracy"],
            "F1-Weighted": classify_report["weighted avg"]["f1-score"],
            "F1-Avg": classify_report["macro avg"]["f1-score"]
        }
    json.dump(logs, open(os.path.join(folder_name, "logs.json"), 'w+'))

