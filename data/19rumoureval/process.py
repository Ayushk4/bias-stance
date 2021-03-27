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
    labels_train = json.load(open("scrapped_full/train-key.json", 'r'))
    labels_dev = json.load(open("scrapped_full/dev-key.json", 'r'))
    event_files = ["scrapped_full/rumoureval-2019-training-data/twitter-english/" + f 
                    for f in os.listdir("scrapped_full/rumoureval-2019-training-data/twitter-english")] + \
                        ['scrapped_full/rumoureval-2019-training-data/' + x for x in ["reddit-dev-data",  "reddit-training-data"]]
    tree_paths = [e + "/" + x for e in event_files for x in os.listdir(e)]

    tweet_paths = {tree_path+"/"+type_tweet+"/"+f: tree_path+"/source-tweet/"+tree_path.split("/")[-1]+".json"
                    for tree_path in tree_paths
                        for type_tweet in ["source-tweet", "replies"] 
                            for f in os.listdir(tree_path + "/" +type_tweet)
                    }

    id2path = {p.split("/")[-1].split(".")[0]: (p, target) for p,target in tweet_paths.items()}

    def get_dataset(labels):
        dataset = []
        cnt, cnt_src, cnt_src_neg = 0, 0, 0
        for id_, label in labels.items():
            path_source, path_target = id2path[id_]
            if "reddit" in path_source:
                this_target = json.load(open(path_target, 'r'))['data']['children'][0]['data']['title']
                this_json = json.load(open(path_source, 'r'))['data']
                if 'source-tweet' in path_source:
                    cnt_src += 1
                    this_text = this_target 
                else:
                    if 'body' in this_json.keys():
                        this_text = this_json['body']
                    else:
                        cnt += 1
                        this_text = "N///A"
            else:
                this_text = json.load(open(path_source, 'r'))['text']
                this_target = json.load(open(path_target, 'r'))['text']
            dataset.append({"tweet_id": id_, "stance": label,
                            "text": this_text, 'target': this_target
                        })
        print(len(dataset), len(labels), cnt, cnt_src, cnt_src_neg)
        return dataset

    return get_dataset(labels_train['subtaskaenglish']), get_dataset(labels_dev['subtaskaenglish'])

def test():
    labels_test = json.load(open("scrapped_full/final-eval-key.json", 'r'))
    event_files = ["scrapped_full/rumoureval-2019-test-data/twitter-en-test-data/" + f 
                    for f in os.listdir("scrapped_full/rumoureval-2019-test-data/twitter-en-test-data")] + \
                        ['scrapped_full/rumoureval-2019-test-data/' + x for x in ["reddit-test-data"]]
    tree_paths = [e + "/" + x for e in event_files for x in os.listdir(e)]

    tweet_paths = {tree_path+"/"+type_tweet+"/"+f: tree_path+"/source-tweet/"+tree_path.split("/")[-1]+".json"
                    for tree_path in tree_paths
                        for type_tweet in ["source-tweet", "replies"] 
                            for f in os.listdir(tree_path + "/" +type_tweet)
                    }

    id2path = {p.split("/")[-1].split(".")[0]: (p, target) for p,target in tweet_paths.items()}

    def get_dataset(labels):
        dataset = []
        cnt, cnt_src, cnt_src_neg = 0, 0, 0
        for id_, label in labels.items():
            path_source, path_target = id2path[id_]
            if "reddit" in path_source:
                this_target = json.load(open(path_target, 'r'))['data']['children'][0]['data']['title']
                this_json = json.load(open(path_source, 'r'))['data']
                if 'source-tweet' in path_source:
                    cnt_src += 1
                    this_text = this_target 
                else:
                    if 'body' in this_json.keys():
                        this_text = this_json['body']
                    else:
                        cnt += 1
                        this_text = "N///A"
            else:
                this_text = json.load(open(path_source, 'r'))['text']
                this_target = json.load(open(path_target, 'r'))['text']
            dataset.append({"tweet_id": id_, "stance": label,
                            "text": this_text, 'target': this_target
                        })
        print(len(dataset), len(labels), cnt, cnt_src, cnt_src_neg)
        
        return dataset
    return get_dataset(labels_test['subtaskaenglish'])

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
        if new_data['text'] == []:
            new_data['text'] = ['']
        new_data['target'] = pre_process_single(datapoint['target'], 0)[:99]
        if new_data['target'] == []:
            new_data['target'] = ['']
        new_data['split'] = split
        preproced_dataset.append(new_data)

fo = open("data.json", "w+")
json.dump(preproced_dataset, fo, indent=2)
fo.close()

