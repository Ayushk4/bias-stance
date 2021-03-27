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


dataset = {l.split(',')[0]: l.strip().split(',')[1:]
            for l in open("all_data_tweet_id.txt", 'r').readlines()[1:]
        }

scrapped_tweets = [x.split('.')[0] for x in os.listdir('scrapped_full')]
prep_dataset = []
for key in scrapped_tweets:
    labels = dataset[key]
    if labels[0] == 'Hilary Clinton':
        this_targets = (labels[2], labels[0])
        this_labels = (labels[3], labels[1])
    else:
        this_targets = (labels[0], labels[2])
        this_labels = (labels[1], labels[3])

    raw_text = json.load(open("scrapped_full/" + key + ".json", 'r'))['full_text']
    prep_dataset.append({'tweet_id': key,
        'target': this_targets,
        'text': pre_process_single(raw_text, 0),
        'stance': this_labels,
        'split': labels[4]
    })


fo = open("data.json", "w+")
json.dump(prep_dataset, fo, indent=2)
fo.close()


