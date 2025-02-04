# Ben Kabongo
# Feb 2025

# Absa: Utils


import nltk
import numpy as np
import pandas as pd
import random
import re
import time
import torch

from nltk.stem.snowball import EnglishStemmer
from nltk.stem import WordNetLemmatizer
from typing import *

from enums import TaskType


def create_res_df_from_dict(res_dict, task_type):
    res_data = []

    for split, res_split in res_dict.items():
        if res_split == {}:
            continue

        if split == "test":
            data = {}
            data["Split"] = "test"
            data["Epoch"] = None
            if task_type is TaskType.T2A:
                data["F1"] = res_split["f1"]
                data["P"] = res_split["precision"]
                data["R"] = res_split["recall"]
                data["#Examples"] = res_split["n_examples"]
                data["#Pred"] = res_split["n_pred"]
                data["#True"] = res_split["n_true"]
            else:
                data["Bleu"] = res_split["BLEU"]["bleu"]
                data["Meteor"] = res_split["METEOR"]["meteor"]
                data["Rouge1"] = res_split["ROUGE"]["rouge1"]
                data["Rouge2"] = res_split["ROUGE"]["rouge2"]
                data["RougeL"] = res_split["ROUGE"]["rougeL"]
                data["BS P"] = res_split["BERTScore"]["precision"]
                data["BS R"] = res_split["BERTScore"]["recall"]
                data["BS F1"] = res_split["BERTScore"]["f1"]
                data["#Examples"] = res_split["n_examples"]
            res_data.append(data)
            
        else:
            for sub_split, res_sub_split in res_split.items():
                if res_sub_split == {}:
                    continue

                if task_type is TaskType.T2A:
                    for i in range(len(res_sub_split["f1"])):
                        data = {}
                        data["Split"] = f"{split}-{sub_split}"
                        data["Epoch"] = i + 1
                        data["F1"] = res_sub_split["f1"][i]
                        data["P"] = res_sub_split["precision"][i]
                        data["R"] = res_sub_split["recall"][i]
                        data["#Examples"] = res_sub_split["n_examples"][i]
                        data["#Pred"] = res_sub_split["n_pred"][i]
                        data["#True"] = res_sub_split["n_true"][i]
                        res_data.append(data)
                else:
                    for i in range(len(res_sub_split["BLEU"]["bleu"])):
                        data = {}
                        data["Split"] = f"{split}-{sub_split}"
                        data["Epoch"] = i + 1
                        data["Bleu"] = res_sub_split["BLEU"]["bleu"][i]
                        data["Meteor"] = res_sub_split["METEOR"]["meteor"][i]
                        data["Rouge1"] = res_sub_split["ROUGE"]["rouge1"][i]
                        data["Rouge2"] = res_sub_split["ROUGE"]["rouge2"][i]
                        data["RougeL"] = res_sub_split["ROUGE"]["rougeL"][i]
                        data["BS P"] = res_sub_split["BERTScore"]["precision"][i]
                        data["BS R"] = res_sub_split["BERTScore"]["recall"][i]
                        data["BS F1"] = res_sub_split["BERTScore"]["f1"][i]
                        data["#Examples"] = res_sub_split["n_examples"][i]
                        res_data.append(data)

    return pd.DataFrame(res_data)


def update_infos(train_infos, test_infos, train_loss_infos, train_epoch_infos, test_epoch_infos):
    train_infos["loss"].append(train_loss_infos["loss"])

    for metric in train_epoch_infos:
        if isinstance(train_epoch_infos[metric], dict):
            if metric not in train_infos:
                train_infos[metric] = {}
                test_infos[metric] = {}

            for k in train_epoch_infos[metric]:
                if k not in train_infos[metric]:
                    train_infos[metric][k] = []
                    test_infos[metric][k] = []
                train_infos[metric][k].append(train_epoch_infos[metric][k])
                test_infos[metric][k].append(test_epoch_infos[metric][k])

        else:
            if metric not in train_infos:
                train_infos[metric] = []
                test_infos[metric] = []
            train_infos[metric].append(train_epoch_infos[metric])
            test_infos[metric].append(test_epoch_infos[metric])

    return train_infos, test_infos


def set_seed(args):
    args.time_id = int(time.time())
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed_all(args.random_state)
    torch.backends.cudnn.deterministic = True


def delete_punctuation(text: str) -> str:
    punctuation = r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\n\t]"
    text = re.sub(punctuation, " ", text)
    text = re.sub('( )+', ' ', text)
    return text


def delete_stopwords(text: str) -> str:
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return ' '.join([w for w in text.split() if w not in stop_words])


def delete_non_ascii(text: str) -> str:
    return ''.join([w for w in text if ord(w) < 128])


def delete_digit(text: str) -> str:
    return re.sub('[0-9]+', '', text)


def first_line(text: str) -> str:
    return re.split(r'[.!?]', text)[0]


def last_line(text: str) -> str:
    if text.endswith('\n'): text = text[:-2]
    return re.split(r'[.!?]', text)[-1]


def stem(text: str) -> str:
    stemmer = EnglishStemmer()
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def lemmatize(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text


def preprocess_text(text: str, args: Any, max_length: int=-1) -> str:
    text = str(text).strip()
    if args.lower_flag: text = text.lower()
    if args.delete_punctuation_flag: text = delete_punctuation(text)
    if args.delete_stopwords_flag: text = delete_stopwords(text)
    if args.delete_non_ascii_flag: text = delete_non_ascii(text)
    if args.first_line_flag: text = first_line(text)
    if args.last_line_flag: text = last_line(text)
    if args.stem_flag: text = stem(text)
    if args.lemmatize_flag: text = lemmatize(text)
    if max_length > 0:
        text = str(text).strip().split()
        if len(text) > max_length:
            text = text[:max_length - 1] + ["..."]
        text = " ".join(text)
    return text