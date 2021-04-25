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


def readcsv(filename, delim, split):
    dataset = []
    with open(filename, 'r', encoding='utf-8', errors="surrogateescape") as f:
        reader = csv.reader(f, delimiter=delim)
        for i, row in enumerate(reader):
            if split == 'test':
                dataset.append([row[0], row[2], row[1], row[3]])
            else:
                if row[1] == 'Donald Trump':
                    continue
                dataset.append([split + "_" + str(i)] + row[:3])
    return dataset[1:]

train_dataset = readcsv("scrapped_full/train.csv", ',', 'train')
test_dataset = readcsv("scrapped_full/test.csv", ',', 'test')

prep_dataset = []

def append_dataset(dataset, split):
    val_num = len(dataset) * 0.75 
    for i, data in enumerate(dataset):
        assert len(data) == 4
        if split == "train" and i > val_num:
            split = "valid"
        prep_dataset.append({'tweet_id': data[0],
                            'text': pre_process_single(data[1], ''),
                            'target': data[2],
                            'split': split,
                            'stance': data[3]
                        })

append_dataset(train_dataset, 'train')
append_dataset(test_dataset, "test")

fo = open("data.json", "w+")
json.dump(prep_dataset, fo, indent=2)
fo.close()

