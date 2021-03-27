# Encryption Debate Dataset

1. Please obtain the encryption debate csv dataset annotation file from [this GitHub repository](https://github.com/aseelad/The-Encryption-Debate) and save it with the filename `The Encryption Debate Dataset.csv`

2. To extract the tweets content please register your application on the twitter developer API and download the tweets. Save all the tweets in a single folder named `scrapped_full/` with each file named in the format `<tweet_id>.json` where tweet_id is a 17-20 digit tweet id.

3. Prepare the dataset - `python3 process.py`. Output will be `data.json`.

