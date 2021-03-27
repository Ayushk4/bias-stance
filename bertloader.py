import torch
import os, json, random, sys
from params import params
import numpy as np
from transformers import AutoTokenizer

np.random.seed(params.seed)
random.seed(params.seed)
torch.manual_seed(params.seed)

MAX_LEN = 0

basepath = "/".join(os.path.realpath(__file__).split('/')[:-1])


FOLDER_NAME = {"16se" : "16semeval",
                "wtwt": "wtwt",
                "enc" : "encryption",
                "17re": "17rumoureval",
                "19re": "19rumoureval",
                "mt1" : "mt",
                "mt2" : "mt"
            }[params.dataset_name]

DATA_PATH = os.path.join(basepath, 'data', FOLDER_NAME, 'data.json')


class StanceDataset:
    def __init__(self):
        self.stance2id =  { "16se": {'AGAINST': 0, 'NONE':      1, 'FAVOR':   2},
                            "wtwt": {'comment': 0, 'unrelated': 1, 'support': 2, 'refute': 3},
                            "enc":  {'FOR':     0, 'NEUTRAL':   1},
                            "17re": {'deny':    0, 'support':   1, 'comment': 2, 'query':  3},
                            "19re": {'deny':    0, 'support':   1, 'comment': 2, 'query':  3},
                            "mt1":  {'AGAINST': 0, 'NONE':      1, 'FAVOR':   2},
                            "mt2":  {'AGAINST': 0, 'NONE':      1, 'FAVOR':   2}
                    }[params.dataset_name]
        self.id2stance = {v: k for k,v in self.stance2id.items()}
        print(self.stance2id, "||", self.id2stance)

        train, eval_set = self.load_dataset(DATA_PATH)

        if params.bert_type == "vinai/bertweet-base":
            self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type, normalization=True)
        else:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
        new_special_tokens_dict = {"additional_special_tokens": ["<number>", "<money>", "<user>"]}
        self.bert_tokenizer.add_special_tokens(new_special_tokens_dict)
        print("Loaded Bert Tokenizer")

        if params.dummy_run == True:
            self.train_dataset, self.criterion_weights = self.batched_dataset([train[0]] * 2)
            self.eval_dataset, _ = self.batched_dataset([train[0]] * 2)
        else:
            print("Train_dataset:", end= " ")
            self.train_dataset, self.criterion_weights = self.batched_dataset(train)
            print("Eval_dataset:", end= " ")
            self.eval_dataset, _ = self.batched_dataset(eval_set)

        # self.criterion_weights = torch.tensor(self.criterion_weights.tolist()).to(params.device)
        # print("Training loss weighing = ", self.criterion_weights)

    def cross_valiation_split(self, train):
        assert params.test_mode != True, "Cross Validation cannot be done while testing"
        split = len(train) // 5
        valid_num = params.cross_valid_num
        if valid_num == 4:
            split *= 4
            train, valid = train[:split], train[split:]
        elif valid_num == 0:
            train, valid = train[split:], train[:split]
        else:
            train, valid = train[:(split * valid_num)] + train[(split * (valid_num+1)):],   train[(split * valid_num):(split * (valid_num+1))]
        return train, valid

    def load_dataset(self, path):
        # Load the dataset
        full_dataset = json.load(open(path, "r"))

        # Split the dataset
        train, valid, test = [], [], []
        for data in full_dataset:
            if params.dataset_name == "wtwt":
                if data['merger'] == "DIS_FOX":
                    # For Dis fox merge this only appears as a test set in all settings.
                    if params.target_merger == "DIS_FOX":
                        test.append(data)
                else:
                    assert data['merger'] in ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX']
                    if params.target_merger == data['merger']:
                        test.append(data)
                    else:
                        train.append(data)
            else:
                if params.dataset_name == 'enc':
                    if data['stance'].lower() == 'against':
                        continue
                if data["split"].lower() == "train":
                    train.append(data)
                elif data['split'].lower() in ["valid", "dev"]:
                    valid.append(data)
                elif data['split'].lower() == "test":
                    test.append(data)
                else:
                    print(data['split'].lower())
                    raise "Unknown split: " + data['split']

        print("Length of train, valid, test:", len(train), len(valid), len(test))
        if params.test_mode:
            train += valid
            valid = []
            assert len(train) != 0 and len(test) != 0
            eval_set = test
        else:
            if params.dataset_name not in ["wtwt", 'enc']:
                assert len(valid) != 0
            else:
                assert len(valid) == 0
                train, valid = self.cross_valiation_split(train)
            eval_set = valid
        print("Length of train, eval_set:", len(train), len(eval_set))
        print("Before shuffling train[0] = ", train[0]["tweet_id"], end=" | ")
        random.shuffle(train)
        print("After shuffling train[0] = ", train[0]["tweet_id"])

        return train, eval_set

    def batched_dataset(self, unbatched): # For batching full or a part of dataset.
        dataset = []
        if params.dataset_name == "enc":
            criterion_weights = np.zeros(2) + 0.0000001 # 2 labels
        elif params.dataset_name in ["16se", "mt1", "mt2"]:
            criterion_weights = np.zeros(3) + 0.0000001 # 3 labels
        else:
            criterion_weights = np.zeros(4) + 0.0000001 # 4 labels 

        idx = 0
        num_data = len(unbatched)

        while idx < num_data:
            batch_text = []
            stances = []
            
            for single_tweet in unbatched[idx:min(idx+params.batch_size, num_data)]:
                if params.dataset_name == "mt1":
                    this_stance_ids = self.stance2id[single_tweet["stance"][0]]
                elif params.dataset_name == "mt2":
                    this_stance_ids = self.stance2id[single_tweet["stance"][1]]
                else:
                    this_stance_ids = self.stance2id[single_tweet["stance"]]
                criterion_weights[this_stance_ids] += 1
                stances.append(this_stance_ids)

                this_tweet = single_tweet['text']

                if params.notarget:
                    this_target = ""
                else:
                    if params.dataset_name == 'mt1':
                        this_target = single_tweet["target"][0]
                    elif params.dataset_name == 'mt2':
                        this_target = single_tweet["target"][1]
                    else:
                        this_target = single_tweet["target"]
                batch_text.append([this_tweet, this_target])

            tokenized_batch = self.bert_tokenizer.batch_encode_plus(batch_text, pad_to_max_length=True,
                                                                return_tensors="pt", return_token_type_ids=True)

            texts = tokenized_batch['input_ids'].to(params.device)
            stances = torch.LongTensor(stances).to(params.device)
            pad_masks = tokenized_batch['attention_mask'].squeeze(1).to(params.device)
            segment_embed = tokenized_batch['token_type_ids'].to(params.device)

            global MAX_LEN
            MAX_LEN = max(MAX_LEN, texts.shape[1])
            # print("\n", stances, stances.size())
            # print("\n", pad_masks[0, :], pad_masks.size())
            # print("\n", segment_embed[0, :], segment_embed.size())

            b = params.batch_size if (idx + params.batch_size) < num_data else (num_data - idx)
            l = texts.size(1)
            assert texts.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP
            assert stances.size() == torch.Size([b])
            assert pad_masks.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP
            assert segment_embed.size() == torch.Size([b, l]) # Maxlen = 63 + 2 for CLS and SEP

            dataset.append((texts, stances, pad_masks, segment_embed))
            idx += params.batch_size

        print("num_batches=", len(dataset), " | num_data=", num_data)
        criterion_weights = np.sum(criterion_weights)/criterion_weights
        print(MAX_LEN)
        return dataset, criterion_weights/np.sum(criterion_weights)

if __name__ == "__main__":
    dataset = StanceDataset()
    print("Train_dataset Size =", len(dataset.train_dataset),
            "Eval_dataset Size =", len(dataset.eval_dataset))
    print(len(dataset.train_dataset))#[0])
    print(dataset.train_dataset[-1])
    #print(len(dataset.hard_dataset))
    import os
    os.system("nvidia-smi")
    print(MAX_LEN)
