# Will-They-Won't-They Dataset

1. Please obtain the wtwt dataset annotation (tweetid-stance-labels) file from [this GitHub repository](https://github.com/cambridge-wtwt/acl2020-wtwt-tweets) and save it with the filename `wtwt_ids.json`

2. To extract the tweets content please register your application on the twitter developer API and download the tweets. Save all the tweets in a single folder named `scrapped_full/` with each file named in the format `<tweet_id>.json` where tweet_id is a 17-20 digit tweet id. Add the desired target sentences for each merger in merger2target.json inside this folder.

3. To prepare the dataset, please set up the dependencies and follow the above two steps. Then execute - `python3 process.py`. For some experiments we perform a extra processing from `extra` branch of this repository. Both the cases output will be in `data.json`.

