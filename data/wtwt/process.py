import json
import os
import sys
import glob

import re
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

id2text = {}

for fil in glob.glob("scrapped_full/*"):
    fo = open(fil, "r")
    full_tweet = json.load(fo)
    fo.close()

    tweet_id = full_tweet["id_str"]

    txt = pre_process_single(full_tweet["full_text"], full_tweet["id"])
    if len(txt) > 0 and txt != ["<user>"]:
        id2text[tweet_id] = txt
    else: 
        pass
print("Processed all tweets")
fo = open("wtwt_ids.json", "r")
wtwt = json.load(fo)
fo.close()

all_keys = id2text.keys()
wtwt_obtained = []

merger2target = json.load(open('merger2target.json'))

for data in wtwt:
    if data["tweet_id"] in all_keys:
        d = data.copy()
        if d['merger'] == 'FOXA_DIS':
            d['merger'] = 'DIS_FOX'
        assert d["merger"] in ['CVS_AET', 'CI_ESRX', 'AET_HUM', 'ANTM_CI', 'DIS_FOX']
        d["text"] = id2text[data["tweet_id"]]
        d["target"] = merger2target[d["merger"]]
        wtwt_obtained.append(d)

print("Processed all dataset")

fo = open("data.json", "w+")
json.dump(wtwt_obtained, fo, indent=2)
fo.close()
