# RumourEval 2019 Dataset

1. Please obtain the RumourEval 2019 task dataset [annotations](https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools) and save it in the folder `scrapped full` with the name

2. If needed, then extract the tweets content. Inside `scrapped full`, please store in the tweet trees inside `train_dev_trees` and `test_trees` folder.  structured inside - please register your application on the twitter developer API and download the tweets. Save all the tweets in a single file for each split insider `scrapped_full/` with each file named in the format `<split>.csv`.

3. Prepare the dataset - `python3 process.py`. Output will be `data.json`.



