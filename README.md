# Bias Stance

<i>t</i>WT–WT: A Dataset to Assert the Role of Target Entities for Detecting Stance

**Accepted to appear** at NAACL-HLT 2021.

This repository contains the models and code accompanying the paper.

ArXiv Link: `coming-soon`

PDF Link: https://www.aclweb.org/anthology/2021.naacl-main.303.pdf

Poster and Slides: `coming-soon`

## Overview

### Abstract

The stance detection task aims at detecting the stance of a tweet or a text for a target. These targets can be named entities or free-form sentences (claims). Though the task involves reasoning of the tweet with respect to a target, we find that it is possible to achieve high accuracy on several publicly available Twitter stance detection datasets without looking at the target sentence. Specifically, a simple tweet classification model achieved human-level performance on the WT–WT dataset and more than two-third accuracy on various other datasets. We investigate the existence of biases in such datasets to find the potential spurious correlations of sentiment-stance relations and lexical choice associated with the stance category. Furthermore, we propose a new large dataset free of such biases and demonstrate its aptness on the existing stance detection systems. Our empirical findings show much scope for research on the stance detection task and proposes several considerations for creating future stance detection datasets.

![Alt text](https://github.com/Ayushk4/bias-stance/blob/master/images/image-bias-stance.jpg)


## Dependencies

| Dependency                  | Version | Installation Command                                                |
| ----------                  | ------- | ------------------------------------------------------------------- |
| Python                      | 3.8.5   | `conda create --name stance python=3.8` and `conda activate stance` |
| PyTorch, cudatoolkit        | 1.5.0   | `conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch`          |
| Transformers  (HuggingFace) | 3.5.0   | `pip install transformers==3.5.0`     |
| Scikit-learn                | 0.23.1  | `pip install scikit-learn==0.23.1`    |
| scipy                       | 1.5.0   | `pip install scipy==1.5.0`            |
| Ekphrasis                   | 0.5.1   | `pip install ekphrasis==0.5.1`        |
| emoji                       | 0.6.0   | `pip install emoji`                   |
| wandb                       | 0.9.4   | `pip install wandb==0.9.4`            |


## Instructions


### Directory Structure

Following is the structure of the codebase, in case you wish to play around with it.

- `train.py`: Model and training loop.
- `bertloader.py`: Common Dataloader for the 6 datasets.
- `params.py`: Argparsing to enable easy experiments.
- `README.md`: This file :slightly_smiling_face:
- `.gitignore`: N/A
- `data`: Directory to store all datasets
  - `data/wtwt`: Folder for WT–WT dataset
    - `data/wtwt/README.md`: README for setting up WT–WT dataset
    - `data/wtwt/process.py`: Script to set up the WT–WT dataset
  - `data/mt`: Folder for the MT dataset
    - `data/mt/README.md`: README for setting up MT dataset
    - `data/mt/process.py`: Script to set up the MT dataset
  - `data/encryption`: Folder for the Encryption-Debate dataset
    - `data/encryption/README.md`: README for setting up Encryption-Debate dataset
    - `data/encryption/process.py`: Script to set up the Encryption-Debate dataset
  - `data/19rumoureval`: Folder for the RumourEval2019 dataset
    - `data/19rumoureval/README.md`: README for setting up RumourEval2019 dataset
    - `data/19rumoureval/process.py`: Script to set up the RumourEval2019 dataset
  - `data/17rumoureval`: Folder for the RumourEval2017 dataset
    - `data/17rumoureval/README.md`: README for setting up RumourEval2017 dataset
    - `data/17rumoureval/process.py`: Script to set up the RumourEval2017 dataset
  - `data/16semeval`: Folder for the Semeval 2016 dataset
    - `data/16semeval/README.md`: README for setting up Semeval 2016 dataset
    - `data/16semeval/process.py`: Script to set up the Semeval 2016 dataset


### 1. Setting up the codebase and the dependencies.

- Clone this repository - `git clone https://github.com/Ayushk4/bias-stance`
- Follow the instructions from the [`Dependencies`](#dependencies) Section above to install the dependencies.
- If you are interested in logging your runs, Set up your wandb - `wandb login`.

### 2. Setting up the datasets.

This codebase supports the 6 datasets considered in our paper.
- Will-They-Won't-They Dataset ([Conforti et al., 2020](https://doi.org/10.18653/v1/2020.acl-main.157)) - `wtwt`
- SemEval 2016 Task 6 Dataset ([Mohammed et al., 2016](https://doi.org/10.18653/v1/S16-1003)) - `16semeval`
- Multi-Target (M-T) Stance Dataset ([Sobhani et al., 2017](https://www.aclweb.org/anthology/E17-2088)) - `mt`
- RumourEval 2017 Task Dataset ([Derczynski et al., 2017](https://doi.org/10.18653/v1/S17-2006)) - `17rumoureval`
- RumourEval 2019 Task Dataset ([Gorrell et al., 2019](https://doi.org/10.18653/v1/S19-2147)) - `19rumoureval`
- Encryption Debate Dataset ([Addawood et al., 2017](https://doi.org/10.1145/3097286.3097288)) - `encryption`

For each `<dataset-name>` set up up inside its respective folder `data/<dataset-name>`. The instruction to set up each `<dataset-name>` can be found inside `data/<dataset-name>/README.md`. After following those steps, the final processed data will stored in a json format `data/<dataset-name>/data.json`, which will be input to our model.

### 3. Training the models.

We experimented with two models

- Target Oblivious Bert

<img src="https://github.com/Ayushk4/bias-stance/blob/master/images/target-oblivious-bert.png" alt="target-oblivious-bert" width="200"/>

- Target Aware Bert

<img src="https://github.com/Ayushk4/bias-stance/blob/master/images/target-aware-bert.png" alt="target-aware-bert" width="200"/>



After following the above steps, move to the basepath for this repository - `cd bias-stance` and recreate the experiments by executing `python3 train.py [ARGS]` where `[ARGS]` are the following:

Required Args:
- dataset_name: The name of dataset to run the experiment on. Possible values are ["16se", "wtwt", "enc", "17re", "19re", "mt1", "mt2"]; Example Usage: `--dataset_name=wtwt`; Type: `str`; This is a required argument.
- target_merger: When dataset is wtwt, this argument is required to tell the target merger. Example Usage: `--target_merger=CVS_AET`; Type: `str`; Valid Arguments: ['CVS_AET', 'ANTM_CI', 'AET_HUM', 'CI_ESRX', 'DIS_FOX'] or not including the argument.
- test_mode: Indicates whether to evaluate on the test in the run; Example Usage: `--test_mode=False`; Type: `str`
- bert_type: A required argument to specify the bert weights to be loaded. Refer [HuggingFace](https://huggingface.co/models). Example Usage: `--bert_type=bert-base-cased`; Type: `str`

Optional Args:
- seed: The seed for the current run. Example Usage: `--seed=1`; Type: `int`
- cross_validation_num: A helper input for cross validation in wtwt and enc datasets. Example Usage: `--cross_valid_num=1`; Type: `int`
- batch_size: The batch size. Example Usage: `--batch_size=16`; Type: `int`
- lr: Learning Rate. Example Usage: `--lr=1e-5`; Type: `float`
- n_epochs: Number of epochs. Example Usage: `--n_epochs=5`; Type: `int`
- dummy_run: Include `--dummy_run` flag to perform a dummy run with a single trainign and validation batch.
- device: CPU or CUDA. Example Usage: `--device=cpu`; Type: `str`
- wandb: Include `--wandb` flag if you want your runs to be logged to wandb.
- notarget: Include `--notarget` flag if you want the model to be target oblivious.


## Results

### WT–WT

| Model            | CVS_AET F1 | CI_ESRX F1 | ANTM_CI F1 | AET_HUM F1 | Average F1 | Weighted F1 | DIS_FOX F1 |
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ---------- |
| Bert (no-target) | 0.673      | 0.703      | 0.745      | 0.759      | 0.720      | 0.720       | 0.347      |
| Human Upperbound | 0.753      | 0.712      | 0.744      | 0.737      | 0.736      | 0.743       | N/A        |
| Bert (target)    | 0.668      | 0.709      | 0.746      | 0.756      | 0.720      | 0.719       | 0.433      |
| Random guessing  | 0.222      | 0.237      | 0.231      | 0.236      | 0.230      | 0.232       | 0.201      |
| Majority guessing| 0.162      | 0.139      | 0.155      | 0.134      | 0.151      | 0.148       | 0.161      |

### SemEval 2016 Dataset

| Model            | Accuracy | F1 Weighted | F1 Macro |
| ---------------- | -------- | ----------- | -------- |
| Bert (no target) | 0.708    | 0.711       | 0.675    |
| Bert (target)    | 0.738    | 0.737       | 0.695    |
| Majority Class   | 0.572    | 0.416       | 0.243    |
| Random           | 0.333    | 0.353       | 0.313    |

### Multi-Target Dataset

| Model            | Accuracy | F1 Weighted | F1 Macro |
| ---------------- | -------- | ----------- | -------- |
| Bert (no target) | 0.675    | 0.673       | 0.654    |
| Bert (target)    | 0.691    | 0.681       | 0.657    |
| Majority Class   | 0.419    | 0.247       | 0.197    |
| Random           | 0.333    | 0.336       | 0.331    |

### RumourEval 2017 Dataset

| Model            | Accuracy | F1 Weighted | F1 Macro |
| ---------------- | -------- | ----------- | -------- |
| Bert (no target) | 0.783    | 0.766       | 0.543    |
| Bert (target)    | 0.769    | 0.760       | 0.543    |
| Majority Class   | 0.742    | 0.632       | 0.213    |
| Random           | 0.250    | 0.310       | 0.189    |

### RumourEval 2019 Dataset

| Model            | Accuracy | F1 Weighted | F1 Macro |
| ---------------- | -------- | ----------- | -------- |
| Bert (no target) | 0.840    | 0.821       | 0.577    |
| Bert (target)    | 0.836    | 0.829       | 0.604    |
| Majority Class   | 0.808    | 0.722       | 0.223    |
| Random           | 0.250    | 0.329       | 0.171    |

### Encryption Debate Dataset

| Model            | Accuracy | F1 Weighted | F1 Macro |
| ---------------- | -------- | ----------- | -------- |
| Bert (no target) | 0.916    | 0.903       | 0.778    |
| Bert (target)    | 0.907    | 0.894       | 0.755    |
| Majority Class   | 0.863    | 0.801       | 0.464    |
| Random           | 0.500    | 0.576       | 0.424    |


## Trained Models

| Model               | Accuracy | F1-Wtd      | F1-Macro | Batch | lr   | Epoch | Model Weights |
| ----------------    | -------- | ----------- | -------- | ----- | ---- | ----- | ------------- |
| AET_HUM notarget    | 0.767    | 0.768       | 0.759    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_AET_HUM_notarget.pt)
| AET_HUM target      | 0.765    | 0.767       | 0.756    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_AET_HUM_target.pt)
| ANTM_CI notarget    | 0.786    | 0.788       | 0.745    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_ANTM_CI_notarget.pt)
| ANTM_CI target      | 0.784    | 0.786       | 0.746    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_ANTM_CI_target.pt)
| CI_ESRX notarget    | 0.727    | 0.730       | 0.703    | 16    | 3e-6 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_CI_ESRX_notarget.pt)
| CI_ESRX target      | 0.732    | 0.734       | 0.709    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_CI_ESRX_target.pt)
| CVS_AET notarget    | 0.715    | 0.713       | 0.673    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_CVS_AET_notarget.pt)
| CVS_AET target      | 0.709    | 0.711       | 0.668    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_CVS_AET_target.pt)
| DIS_FOX notarget    | 0.502    | 0.442       | 0.347    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_DIS_FOX_notarget.pt)
| DIS_FOX target      | 0.545    | 0.497       | 0.433    | 16    | 3e-6 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/wtwt_DIS_FOX_target.pt)
| SemEval16 notarget  | 0.708    | 0.711       | 0.675    | 16    | 1e-5 | 2     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/16se_notarget.pt)
| SemEval16 target    | 0.738    | 0.737       | 0.695    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/16se_target.pt)
| Multitarget notarget| 0.675    | 0.673       | 0.654    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance)
| Multitarget target  | 0.691    | 0.681       | 0.657    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance)
| RumourEval17 notarget| 0.783   | 0.766       | 0.543    | 16    | 3e-6 | 10    | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/17re_notarget.pt)
| RumourEval17 target | 0.769    | 0.760       | 0.543    | 16    | 3e-6 | 10    | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/17re_target.pt)
| RumourEval19 notarget| 0.840   | 0.821       | 0.577    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/19re_notarget.pt)
| RumourEval19 target | 0.836    | 0.829       | 0.604    | 16    | 1e-5 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/19re_target.pt)
| Encryption notarget | 0.916    | 0.903       | 0.778    | 16    | 3e-6 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/enc_notarget.pt)
| Encryption target   | 0.907    | 0.894       | 0.755    | 16    | 3e-6 | 5     | [Link](https://github.com/Ayushk4/bias-stance/releases/download/v0.0/enc_target.pt)

## Citation

- Authors: Ayush Kaushal, Avirup Saha and Niloy Ganguly
- Code base written by Ayush Kaushal
- NAACL 2021 Proceedings

Please Cite our paper if you find the codebase useful:

```
@inproceedings{kaushal2020stance,
          title={tWT–WT: A Dataset to Assert the Role of Target Entities for Detecting Stance},
          author={Kaushal, Ayush and Saha, Avirup and Ganguly, Niloy} 
          booktitle={Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2021)},
          year={2021}
        }
```



## Miscellanous

- You may contact us by opening an issue on this repo and/or mailing to the first author - `<this_github_username> [at] gmail.com` Please allow 2-3 days of time to address the issue.

- The codebase has been written from scratch, but was inspired from many others [1](https://github.com/jackroos/VL-BERT) [2](https://propaganda.qcri.org/fine-grained-propaganda-emnlp.html) [3](https://github.com/prajwal1210/Stance-Detection-in-Web-and-Social-Media)

- License: MIT


