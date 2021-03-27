import json, os, sys, re, glob, csv

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

EMOJI_PATTERN = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         "]+", flags=re.UNICODE)

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone',
        'time', 'url', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},

    fix_html=True,
    segmenter="twitter",
    corrector="twitter",

    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=True,

    tokenizer=SocialTokenizer(lowercase=False).tokenize,
    dicts=[emoticons]
)

REMOVE_TAGS = [
    "<emphasis>", "<kiss>", "<repeated>", "<laugh>", "<allcaps>",
    "</allcaps>", "<angel>", "<elongated>", "<tong>", "<annoyed>",
    "<censored>", "<happy>", "<percent>", "<wink>",
    "<headdesk>", "<surprise>", "<date>", "<time>", "<url>",
    "<sad>", "<email>", "<phone>", "<hashtag>", "</hashtag>"
    ]

ADD_TO_GLOVE = ["<number>", "<money>"]

# Try removing punctuations as well

def train_dev():
    labels_train = json.load(open("scrapped_full/train.json", 'r'))
    labels_dev = json.load(open("scrapped_full/dev.json", 'r'))

    event_files = ["scrapped_full/train_dev_trees/" + f
                    for f in os.listdir("scrapped_full/train_dev_trees")
                ]
    tree_paths = [e + "/" + x for e in event_files for x in os.listdir(e)]

    tweet_paths = {tree_path+"/"+type_tweet+"/"+f: tree_path+"/source-tweet/"+tree_path.split("/")[-1]+".json"
                    for tree_path in tree_paths
                        for type_tweet in ["source-tweet", "replies"] 
                            for f in os.listdir(tree_path + "/" +type_tweet)
                    }

    id2path = {p.split("/")[-1].split(".")[0]: (p, target) for p,target in tweet_paths.items()}

    train_dataset = []
    for id_, label in labels_train.items():
        train_dataset.append({"tweet_id": id_,
                            "stance": label,
                            "text": json.load(open(id2path[id_][0], 'r'))['text'],
                            "target": json.load(open(id2path[id_][1], 'r'))['text']
                        })

    dev_dataset = []
    for id_, label in labels_dev.items():
        dev_dataset.append({"tweet_id": id_,
                            "stance": label,
                            "text": json.load(open(id2path[id_][0], 'r'))['text'],
                            "target": json.load(open(id2path[id_][1], 'r'))['text']
                        })

    return train_dataset, dev_dataset


def test():
    labels_test = json.load(open("scrapped_full/test.json", 'r'))

    tree_paths = ["scrapped_full/test_trees/" + f
                    for f in os.listdir("scrapped_full/test_trees")
                ]

    tweet_paths = {tree_path+"/"+type_tweet+"/"+f: tree_path+"/source-tweet/"+tree_path.split("/")[-1]+".json"
                    for tree_path in tree_paths
                        for type_tweet in ["source-tweet", "replies"] 
                            for f in os.listdir(tree_path + "/" +type_tweet)
                    }

    id2path = {p.split("/")[-1].split(".")[0]: (p, target) for p,target in tweet_paths.items()}

    test_dataset = []
    for id_, label in labels_test.items():
        test_dataset.append({"tweet_id": id_,
                            "stance": label,
                            "text": json.load(open(id2path[id_][0], 'r'))['text'],
                            "target": json.load(open(id2path[id_][1], 'r'))['text']
                        })
    return test_dataset

train, dev = train_dev()
test = test()


def pre_process_single(tweet, t_id):
    tweet_toked_text = []
    de_emojified_text = tweet.encode('ascii', 'ignore').decode('ascii')
    de_emojified_text = re.sub(r"\$", " $ ", de_emojified_text)
    company_normalize_text = EMOJI_PATTERN.sub(r' ', de_emojified_text)

    tokens = text_processor.pre_process_doc(company_normalize_text)
    for token in tokens:
        if token in REMOVE_TAGS:
            continue
        tweet_toked_text.append(token)

    return tweet_toked_text


preproced_dataset = []
for dataset, split in [(train, 'train'), (dev, 'dev'), (test, 'test')]:
    for datapoint in dataset:
        new_data = datapoint.copy()
        new_data['text'] = pre_process_single(datapoint['text'], 0)[:99]
        new_data['target'] = pre_process_single(datapoint['target'], 0)[:99]
        new_data['split'] = split
        preproced_dataset.append(new_data)

fo = open("data.json", "w+")
json.dump(preproced_dataset, fo, indent=2)
fo.close()

