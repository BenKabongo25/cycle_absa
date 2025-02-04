# Ben Kabongo
# Feb 2025

# Absa: Data

import ast
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from typing import *

from annotations_text import AnnotationsTextFormerBase
from enums import TaskType, AbsaTupleType
from prompts import Prompter
from utils import preprocess_text


class T5ABSADataset(Dataset):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        annotations_text_former: AnnotationsTextFormerBase,
        prompter: Prompter,
        data_df: pd.DataFrame,
        args: Any,
        task_type: Optional[TaskType]=None,
        split_name: str=""
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.annotations_text_former = annotations_text_former
        self.prompter = prompter

        self.texts = data_df[args.text_column].tolist()
        if args.annotation_flag:
            self.annotations = data_df[args.annotations_column].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            ).tolist()

        self.args = args
        self.task_type = task_type if task_type is not None else self.args.task_type
        self.split_name = split_name

        self.input_texts = []
        self.target_texts = []
        self.references_idx = []

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        self._build()

    def __len__(self):
        return len(self.texts)

    def _build(self):
        desc = "Prepare data" if not self.split_name else self.split_name
        for idx in tqdm(range(len(self)), desc, colour="green"):
            text = preprocess_text(self.texts[idx], self.args)
            
            formatted_annotations = []
            annotations_text = None
            if self.args.annotation_flag:
                annotations = self.annotations[idx]
                for ann in annotations:
                    if not self.args.annotations_raw_format == self.args.absa_tuple.value:
                        ann = AbsaTupleType.format_annotations(
                            ann, self.args.annotations_raw_format, self.args.absa_tuple.value
                        )
                    ann = tuple([preprocess_text(t.strip(), self.args) for t in ann]) 
                    formatted_annotations.append(ann)
                self.annotations[idx] = formatted_annotations
                annotations_text = self.annotations_text_former.multiple_annotations_to_text(formatted_annotations)

            if self.task_type is TaskType.T2A:
                input_text = text
                target_text = annotations_text
            else:
                input_text = annotations_text
                target_text = text

            input_text = self.prompter.get_prompt(
                task_type=self.task_type,
                text=input_text, 
                annotations=formatted_annotations
            )
            
            input = self.tokenizer(
                input_text, 
                max_length=self.args.max_input_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            labels = None
            if self.args.annotation_flag:
                target = self.tokenizer(
                    target_text, 
                    max_length=self.args.max_target_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                labels = target["input_ids"]
                labels[labels == self.tokenizer.pad_token_id] = -100

            self.input_texts.append(input_text)
            self.input_ids.append(input_ids)
            self.attention_masks.append(attention_mask)
            self.target_texts.append(target_text)
            self.labels.append(labels)

        if self.task_type is TaskType.T2A:
            for i in range(len(self)):
                self.references_idx.append([i])
        else:
            for i in range(len(self)):
                refs_idx = [i]
                if self.args.annotation_flag:
                    for j in range(len(self)):
                        if i != j and self.annotations[i] == self.annotations[j]:
                            refs_idx.append(j)
                self.references_idx.append(refs_idx)
        

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]

        annotations = None
        if self.args.annotation_flag:
            annotations = self.annotations[idx]

        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]

        references = [self.target_texts[ref_idx] for ref_idx in self.references_idx[idx]]

        return {
            "input_texts": input_text,
            "target_texts": target_text,
            "annotations": annotations,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "references": references
        }


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        collated_batch[key] = [d[key] for d in batch]
        if isinstance(collated_batch[key][0], torch.Tensor):
            collated_batch[key] = torch.cat(collated_batch[key], 0)
    return collated_batch
        

def get_train_val_test_df(args: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_df = pd.read_csv(args.dataset_path)
    train_df, val_test_df = train_test_split(
        data_df, 
        test_size=args.test_size + args.val_size, 
        random_state=args.random_state
    )
    val_df, test_df = train_test_split(
        val_test_df, 
        test_size=args.test_size / (args.test_size + args.val_size), 
        random_state=args.random_state
    )
    return train_df, val_df, test_df
