import json, os, sys, re, glob

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

all_dataset = [(x.split(',')[0].split('/')[-1], x.split(',')[-1].strip())
                for x in open("The Encryption Debate Dataset.csv").readlines()
                if x.split(',')[-1].strip() in ['FOR', 'NEUTRAL']
        ]

id2stance = {x[0]: [] for x in all_dataset}
for x, y in all_dataset:
    id2stance[x].append(y)

id2stance = {x: list(set(y))[0] for x,y  in id2stance.items() if len(set(y)) == 1}

scrapped_tweets = [x.split('.')[0] for x in os.listdir('scrapped_full')]
prep_dataset = []
train_nums = len(scrapped_tweets) * 75/100

split = "train"
cnt = 0
for key in scrapped_tweets:
    if key not in id2stance:
        continue
    if cnt > train_nums:
        split = "test"
    raw_text = json.load(open("scrapped_full/" + key + ".json", 'r'))['full_text']
    prep_dataset.append({'tweet_id': key,
        'target': "encryption",
        'text': pre_process_single(raw_text, 0),
        'stance': id2stance[key],
        'split': split
    })
    cnt += 1
# assert split == "test"

fo = open("data.json", "w+")
json.dump(prep_dataset, fo, indent=2)
fo.close()


