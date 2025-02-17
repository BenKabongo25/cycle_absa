# Ben Kabongo
# Feb 2025

# Cycle Absa

import argparse
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from annotations_text import AnnotationsTextFormerBase
from data import T5ABSADataset, collate_fn, get_train_val_test_df
from enums import TaskType, AbsaTupleType, AnnotationsTextFormerType
from eval import get_evaluation_scores
from prompts import Prompter
from utils import set_seed, update_infos, create_res_df_from_dict


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))


def empty_cache():
    with torch.no_grad(): 
        torch.cuda.empty_cache()


def evaluate(model, tokenizer, annotations_text_former, dataloader, task_type, args):
    references = []
    predictions = []
    all_inputs = []
    all_outputs = []
    all_annotations = []

    model.eval()

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), "Eval", colour="cyan", total=len(dataloader)):
            torch.cuda.empty_cache()
            
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            
            outputs = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, do_sample=False, 
                max_length=args.max_target_length
            )
            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            input_texts = batch["input_texts"]
            target_texts = batch["target_texts"]
            refs = batch["references"]
            annotations = batch["annotations"]

            references.extend(refs)
            predictions.extend(output_texts)
            all_inputs.extend(input_texts)
            all_outputs.extend(output_texts)
            all_annotations.extend(annotations)

            if batch_idx == 0 and args.verbose:
                log = "=" * 150
                for i in range(len(input_texts)):
                    log += f"\nInput: {input_texts[i]}"
                    log += f"\nReferences: {refs[i][: min(len(refs[i]), 3)]}"
                    log += f"\nAnnotations: {annotations[i]}"
                    log += f"\nOutput: {output_texts[i]}"
                    if task_type is TaskType.T2A:
                        log += (
                            f"\nOutput annotations: "
                            f"{annotations_text_former.multiple_text_to_annotations(output_texts[i])}"
                        )
                    log += "\n"

                print("\n" + log)
                with open(args.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(log)

            del input_ids, attention_mask, outputs, input_texts, target_texts, refs, annotations
            torch.cuda.empty_cache()

    if args.annotation_flag:
        scores = get_evaluation_scores(
            predictions, 
            references, 
            all_annotations, 
            annotations_text_former,
            args,
            task_type
        )
    else:
        scores = {}

    if args.task_type is TaskType.T2A:
        all_output_annotations = []
        for output_text in all_outputs:
            output_annotations = annotations_text_former.multiple_text_to_annotations(output_text)
            all_output_annotations.append(output_annotations)
    else:
        all_output_annotations = all_annotations

    if args.save_eval_results:
        res_df = pd.DataFrame(
            {
                "Input": all_inputs,
                "Annotations": all_output_annotations
            }
        )
        res_df.to_csv(args.eval_res_file_path)

    return scores


def train(model, optimizer, dataloader, args):
    running_loss = .0
    model.train()

    for batch_idx, batch in tqdm(enumerate(dataloader), "Training", colour="cyan", total=len(dataloader)):
        torch.cuda.empty_cache()
        
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        del input_ids, attention_mask, labels, outputs
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    return {"loss": running_loss}


def one_epoch_trainer(
    model, 
    tokenizer, 
    optimizer,
    annotations_text_former, 
    train_dataloader, 
    train_evaluated_dataloader,
    test_dataloader,
    task_type,
    args
):
    train_loss_infos = train(model, optimizer, train_dataloader, args)
    
    train_epoch_infos = evaluate(
        model, tokenizer, annotations_text_former, train_evaluated_dataloader, task_type, args
    )
    test_epoch_infos = evaluate(
        model, tokenizer, annotations_text_former, test_dataloader, task_type, args
    )

    if task_type is TaskType.T2A:
        f1_score = test_epoch_infos["f1"]
        if f1_score > args.best_t2a_f1_score:
            save_model(model, args.save_t2a_model_path)
            args.best_t2a_f1_score = f1_score
    else:
        meteor_score = test_epoch_infos["METEOR"]["meteor"]
        if meteor_score > args.best_a2t_meteor_score:
            save_model(model, args.save_a2t_model_path)
            args.best_a2t_meteor_score = meteor_score

    return train_loss_infos, train_epoch_infos, test_epoch_infos


def trainer(
    model, 
    tokenizer, 
    optimizer,
    annotations_text_former, 
    train_dataloader, 
    train_evaluated_dataloader,
    test_dataloader,
    task_type,
    n_epochs,
    args,
    desc="Training",
    colour="blue",
    train_infos={"loss": [],},
    test_infos={}
):
    progress_bar = tqdm(range(1, 1 + n_epochs), desc, colour=colour)
    for epoch in progress_bar:
        train_loss_infos, train_epoch_infos, test_epoch_infos = one_epoch_trainer(
            model, 
            tokenizer, 
            optimizer,
            annotations_text_former, 
            train_dataloader, 
            train_evaluated_dataloader,
            test_dataloader,
            task_type,
            args
        )
        train_infos, test_infos = update_infos(
            train_infos, test_infos,
            train_loss_infos, train_epoch_infos, test_epoch_infos 
        )

        progress_bar.set_description(
            f"{desc} [{epoch} / {n_epochs}] " +
            f"Loss: train={train_loss_infos['loss']:.4f} "
        )
    
    return train_infos, test_infos


def labeled_trainer(
    model,
    tokenizer,
    optimizer,
    annotations_text_former,
    train_dataloader,
    test_dataloader,
    task_type,
    args
):
    if task_type is TaskType.T2A:
        desc = "T2A Labeled Training"
        colour = "blue"
    else:
        desc = "A2T Labeled Training"
        colour = "yellow"

    train_infos, test_infos = trainer(
        model=model, 
        tokenizer=tokenizer, 
        optimizer=optimizer,
        annotations_text_former=annotations_text_former, 
        train_dataloader=train_dataloader, 
        train_evaluated_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        task_type=task_type,
        n_epochs=args.n_labeled_epochs,
        args=args,
        desc=desc,
        colour=colour
    )

    return train_infos, test_infos


def create_labeled_dataloader(model, tokenizer, annotations_text_former, dataloader, task_type, args):
    all_texts = []
    all_annotations = []

    prompter = dataloader.dataset.prompter
    
    model.eval()
    for batch_idx, batch in tqdm(
        enumerate(dataloader), f"{task_type.value}: Create labeled data", colour="green", 
        total=len(dataloader)
    ):
        empty_cache()
        
        input_texts = batch["input_texts"]
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False, 
            max_length=args.max_target_length
        )
        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if task_type is TaskType.T2A:
            texts = [
                prompter.get_text(task_type=task_type, prompt=input_text) for input_text in input_texts
            ]
            annotations = [
                annotations_text_former.multiple_text_to_annotations(output_text) 
                for output_text in output_texts
            ]
        else:
            texts = output_texts
            annotations = batch["annotations"]

        all_texts.extend(texts)
        all_annotations.extend(annotations)

        if batch_idx == 0 and args.verbose:
            true_input_texts = batch["input_texts"]
            true_target_texts = batch["target_texts"]
            true_annotations = batch["annotations"]
            log = "=" * 150
            log += f"\n\n{task_type.value}: Create labeled data\n"
            for i in range(len(input_texts)):
                log += f"\nInput: {true_input_texts[i]}"
                log += f"\nTarget: {true_target_texts[i]}"
                log += f"\nAnnotations: {true_annotations[i]}"
                log += f"\nOutput: {output_texts[i]}"
                if task_type is TaskType.T2A:
                    log += (
                        f"\nOutput annotations: {annotations[i]}"
                    )
                log += "\n"

            print("\n" + log)
            with open(args.log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log)

    inverse_task_type = TaskType.A2T if task_type is TaskType.T2A else TaskType.T2A
    data_df = pd.DataFrame(
        {args.text_column: all_texts, args.annotations_column: all_annotations}
    )
    dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, 
        data_df, args, task_type=inverse_task_type, 
        split_name=f"Train {task_type.value} -> {inverse_task_type.value}"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )

    return dataloader


def t2a_unlabeled_one_epoch_trainer(
    t2a_model,
    a2t_model,
    tokenizer,
    t2a_optimizer,
    annotations_text_former,
    train_t2a_unlabeled_dataloader,
    train_a2t_unlabeled_dataloader,
    test_t2a_dataloader,
    args,
    t2a_train_infos={"loss": [],},
    t2a_test_infos={}
):
    train_t2a_augmented_dataloader = create_labeled_dataloader(
        model=a2t_model,
        tokenizer=tokenizer,
        annotations_text_former=annotations_text_former,
        dataloader=train_a2t_unlabeled_dataloader,
        task_type=TaskType.A2T,
        args=args
    )

    train_loss_infos, train_epoch_infos, test_epoch_infos = one_epoch_trainer(
        model=t2a_model, 
        tokenizer=tokenizer, 
        optimizer=t2a_optimizer,
        annotations_text_former=annotations_text_former, 
        train_dataloader=train_t2a_augmented_dataloader, 
        train_evaluated_dataloader=train_t2a_unlabeled_dataloader,
        test_dataloader=test_t2a_dataloader,
        task_type=TaskType.T2A,
        args=args
    )

    t2a_train_infos, t2a_test_infos = update_infos(
        t2a_train_infos, t2a_test_infos,
        train_loss_infos, train_epoch_infos, test_epoch_infos 
    )

    return t2a_train_infos, t2a_test_infos


def a2t_unlabeled_one_epoch_trainer(
    a2t_model,
    t2a_model,
    tokenizer,
    a2t_optimizer,
    annotations_text_former,
    train_a2t_unlabeled_dataloader,
    train_t2a_unlabeled_dataloader,
    test_a2t_dataloader,
    args,
    a2t_train_infos={"loss": [],},
    a2t_test_infos={}
):
    train_a2t_augmented_dataloader = create_labeled_dataloader(
        model=t2a_model,
        tokenizer=tokenizer,
        annotations_text_former=annotations_text_former,
        dataloader=train_t2a_unlabeled_dataloader,
        task_type=TaskType.T2A,
        args=args
    )

    train_loss_infos, train_epoch_infos, test_epoch_infos = one_epoch_trainer(
        model=a2t_model, 
        tokenizer=tokenizer, 
        optimizer=a2t_optimizer,
        annotations_text_former=annotations_text_former, 
        train_dataloader=train_a2t_augmented_dataloader, 
        train_evaluated_dataloader=train_a2t_unlabeled_dataloader,
        test_dataloader=test_a2t_dataloader,
        task_type=TaskType.A2T,
        args=args
    )
    
    a2t_train_infos, a2t_test_infos = update_infos(
        a2t_train_infos, a2t_test_infos,
        train_loss_infos, train_epoch_infos, test_epoch_infos 
    )

    return a2t_train_infos, a2t_test_infos


def verbose_results(res_data_df, task_type, args):
    if args.verbose:
        log = "=" * 150
        log += f"\n{task_type.value}\n"
        log += f"\n{res_data_df.head(n=len(res_data_df))}\n"
        print(log)
        with open(args.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log)


def cycle_trainer(
    t2a_model,
    a2t_model,
    tokenizer,
    annotations_text_former,
    train_t2a_labeled_dataloader,
    train_t2a_unlabeled_dataloader,
    train_a2t_labeled_dataloader,
    train_a2t_unlabeled_dataloader,
    test_t2a_dataloader,
    test_a2t_dataloader,
    args
):
    t2a_optimizer = torch.optim.Adam(t2a_model.parameters(), lr=args.lr)
    a2t_optimizer = torch.optim.Adam(a2t_model.parameters(), lr=args.lr)

    args.best_t2a_f1_score = 0.0
    args.best_a2t_meteor_score = 0.0

    t2a_infos = {"test": {}, "labeled": {}, "unlabeled": {}}
    a2t_infos = {"test": {}, "labeled": {}, "unlabeled": {}}

    if args.n_labeled_epochs > 0:
        t2a_train_labeled_infos, t2a_test_labeled_infos = labeled_trainer(
            model=t2a_model,
            tokenizer=tokenizer,
            optimizer=t2a_optimizer,
            annotations_text_former=annotations_text_former,
            train_dataloader=train_t2a_labeled_dataloader,
            test_dataloader=test_t2a_dataloader,
            task_type=TaskType.T2A,
            args=args
        )
        t2a_infos["labeled"]["train"] = t2a_train_labeled_infos
        t2a_infos["labeled"]["eval"] = t2a_test_labeled_infos

        t2a_infos_df = create_res_df_from_dict(t2a_infos, TaskType.T2A)
        t2a_infos_df.to_csv(args.res_t2a_file_path)
        verbose_results(t2a_infos_df, TaskType.T2A, args)

        a2t_train_labeled_infos, a2t_test_labeled_infos = labeled_trainer(
            model=a2t_model,
            tokenizer=tokenizer,
            optimizer=a2t_optimizer,
            annotations_text_former=annotations_text_former,
            train_dataloader=train_a2t_labeled_dataloader,
            test_dataloader=test_a2t_dataloader,
            task_type=TaskType.A2T,
            args=args
        )
        a2t_infos["labeled"]["train"] = a2t_train_labeled_infos
        a2t_infos["labeled"]["eval"] = a2t_test_labeled_infos

        a2t_infos_df = create_res_df_from_dict(a2t_infos, TaskType.A2T)
        a2t_infos_df.to_csv(args.res_a2t_file_path)
        verbose_results(a2t_infos_df, TaskType.A2T, args)

    t2a_progress_bar = tqdm(
        total=args.n_unlabeled_epochs, 
        desc="T2A Unlabeled Trainig", 
        colour="blue", 
        position=1
    )
    a2t_progress_bar = tqdm(
        total=args.n_unlabeled_epochs, 
        desc="A2T Unlabeled Trainig", 
        colour="yellow", 
        position=0
    )

    t2a_train_unlabeled_infos = {"loss": [],}
    t2a_test_unlabeled_infos = {}
    a2t_train_unlabeled_infos = {"loss": [],}
    a2t_test_unlabeled_infos = {}

    for epoch in range(1, 1 + args.n_unlabeled_epochs):
        t2a_train_unlabeled_infos, t2a_test_unlabeled_infos = t2a_unlabeled_one_epoch_trainer(
            t2a_model,
            a2t_model,
            tokenizer,
            t2a_optimizer,
            annotations_text_former,
            train_t2a_unlabeled_dataloader,
            train_a2t_unlabeled_dataloader,
            test_t2a_dataloader,
            args,
            t2a_train_unlabeled_infos,
            t2a_test_unlabeled_infos
        )
        t2a_infos["unlabeled"]["train"] = t2a_train_unlabeled_infos
        t2a_infos["unlabeled"]["eval"] = t2a_test_unlabeled_infos

        train_loss = t2a_train_unlabeled_infos["loss"][-1]
        f1_score = t2a_test_unlabeled_infos["f1"][-1]

        t2a_progress_bar.update(1)
        t2a_progress_bar.set_description(
            f"T2A Unlabeled Trainig [{epoch} / {args.n_unlabeled_epochs}] " +
            f"Loss: train={train_loss:.4f} F1: eval={f1_score:.4f}"
        )
        
        t2a_infos_df = create_res_df_from_dict(t2a_infos, TaskType.T2A)
        t2a_infos_df.to_csv(args.res_t2a_file_path)
        verbose_results(t2a_infos_df, TaskType.T2A, args)

        a2t_train_unlabeled_infos, a2t_test_unlabeled_infos = a2t_unlabeled_one_epoch_trainer(
            a2t_model,
            t2a_model,
            tokenizer,
            a2t_optimizer,
            annotations_text_former,
            train_a2t_unlabeled_dataloader,
            train_t2a_unlabeled_dataloader,
            test_a2t_dataloader,
            args,
            a2t_train_unlabeled_infos,
            a2t_test_unlabeled_infos
        )
        a2t_infos["unlabeled"]["train"] = a2t_train_unlabeled_infos
        a2t_infos["unlabeled"]["eval"] = a2t_test_unlabeled_infos
        
        train_loss = a2t_train_unlabeled_infos["loss"][-1]
        meteor_score = a2t_test_unlabeled_infos["METEOR"]["meteor"][-1]

        a2t_progress_bar.update(1)
        a2t_progress_bar.set_description(
            f"A2T Unlabeled Trainig [{epoch} / {args.n_unlabeled_epochs}] " +
            f"Loss: train={train_loss:.4f} METEOR: eval={meteor_score:.4f}"
        )
        
        a2t_infos_df = create_res_df_from_dict(a2t_infos, TaskType.A2T)
        a2t_infos_df.to_csv(args.res_a2t_file_path)
        verbose_results(a2t_infos_df, TaskType.A2T, args)

    return t2a_infos, a2t_infos


def main(args):
    set_seed(args)

    args.absa_tuple = AbsaTupleType(args.absa_tuple)
    args.annotations_text_type = AnnotationsTextFormerType(args.annotations_text_type)

    annotations_text_former = AnnotationsTextFormerBase.get_annotations_text_former(args)
    prompter = Prompter(args)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.eval_path)
    test_df = pd.read_csv(args.test_path)

    train_t2a_labeled_dataloader = None
    train_a2t_labeled_dataloader = None

    if args.n_labeled_epochs > 0:
        train_labeled_df, train_unlabeled_df = train_test_split(
            train_df, 
            train_size=args.train_labeled_size, 
            random_state=args.random_state
        )

        train_t2a_labeled_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, train_labeled_df, args, task_type=TaskType.T2A,
            split_name=f"Train Labeled T2A"
        )
        train_a2t_labeled_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, train_labeled_df, args, task_type=TaskType.A2T,
            split_name=f"Train Labeled A2T"
        )
        train_t2a_labeled_dataloader = DataLoader(
            train_t2a_labeled_dataset, 
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )
        train_a2t_labeled_dataloader = DataLoader(
            train_a2t_labeled_dataset, 
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )

    else:
        train_unlabeled_df = train_df

    train_t2a_unlabeled_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, train_unlabeled_df, args, task_type=TaskType.T2A,
        split_name=f"Train Unlabeled T2A"
    )
    train_a2t_unlabeled_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, train_unlabeled_df, args, task_type=TaskType.A2T,
        split_name=f"Train Unlabeled A2T"
    )
    train_t2a_unlabeled_dataloader = DataLoader(
        train_t2a_unlabeled_dataset, 
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )
    train_a2t_unlabeled_dataloader = DataLoader(
        train_a2t_unlabeled_dataset, 
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )
    
    val_t2a_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, val_df, args, task_type=TaskType.T2A,
        split_name=f"Eval T2A"
    )
    val_a2t_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, val_df, args, task_type=TaskType.A2T,
        split_name=f"Eval A2T"
    )
    val_t2a_dataloader = DataLoader(
        val_t2a_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )
    val_a2t_dataloader = DataLoader(
        val_a2t_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )

    test_t2a_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, test_df, args, task_type=TaskType.T2A,
        split_name=f"Test T2A"
    )
    test_a2t_dataset = T5ABSADataset(
        tokenizer, annotations_text_former, prompter, test_df, args, task_type=TaskType.A2T,
        split_name=f"Test A2T"
    )
    test_t2a_dataloader = DataLoader(
        test_t2a_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )
    test_a2t_dataloader = DataLoader(
        test_a2t_dataset, 
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=True, num_workers=4
    )

    args.annotations_raw_format = args.absa_tuple.value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(args.exp_dir, "log.txt")
    args.res_t2a_file_path = os.path.join(args.exp_dir, "res_t2a.csv")
    args.res_a2t_file_path = os.path.join(args.exp_dir, "res_a2t.csv")
    args.eval_res_file_path = os.path.join(args.exp_dir, "eval_res.csv")

    args.save_t2a_model_path = os.path.join(args.exp_dir, "t2a_model.pth")
    args.save_a2t_model_path = os.path.join(args.exp_dir, "a2t_model.pth")

    t2a_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    t2a_model.to(args.device)
    if args.load_model:
        load_model(t2a_model, args.save_t2a_model_path)
    
    a2t_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    a2t_model.to(args.device)
    if args.load_model:
        load_model(a2t_model, args.save_a2t_model_path)

    if args.verbose:
        batch = next(iter(test_t2a_dataloader))
        input_texts = batch["input_texts"]
        target_texts = batch["target_texts"]
        annotations = batch["annotations"]
        input_ids = batch["input_ids"]

        log_example = f"Input: {input_texts[0]}"
        log_example += f"\nTarget: {target_texts[0]}"
        log_example += f"\nAnnotations: {annotations[0]}"
        log_example += f"\nInputs size: {input_ids.shape}"
        log = (
            f"Model: {args.model_name_or_path}\n" +
            f"Tokenizer: {args.tokenizer_name_or_path}\n" +
            f"Tuple: {args.absa_tuple}\n" +
            f"Annotations: {args.annotations_text_type}\n" +
            f"Dataset: {args.train_path}\n" +
            f"Device: {device}\n" +
            f"Arguments:\n{args}\n\n" +
            f"Data:\n{test_df.head(5)}\n\n" +
            f"Input-Output example:\n{log_example}\n"
        )
        print("\n" + log)
        with open(args.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(log)

    t2a_infos, a2t_infos = cycle_trainer(
        t2a_model,
        a2t_model,
        tokenizer,
        annotations_text_former,
        train_t2a_labeled_dataloader,
        train_t2a_unlabeled_dataloader,
        train_a2t_labeled_dataloader,
        train_a2t_unlabeled_dataloader,
        val_t2a_dataloader,
        val_a2t_dataloader,
        args
    )

    load_model(t2a_model, args.save_t2a_model_path)
    t2a_model.to(args.device)
    t2a_test_infos = evaluate(
        t2a_model, tokenizer, annotations_text_former, test_t2a_dataloader, TaskType.T2A, args
    )
    t2a_infos["test"] = t2a_test_infos
    t2a_infos_df = create_res_df_from_dict(t2a_infos, TaskType.T2A)
    t2a_infos_df.to_csv(args.res_t2a_file_path)
    verbose_results(t2a_infos_df, TaskType.T2A, args)

    load_model(a2t_model, args.save_a2t_model_path)
    a2t_model.to(args.device)
    a2t_test_infos = evaluate(
        a2t_model, tokenizer, annotations_text_former, test_a2t_dataloader, TaskType.A2T, args
    )
    a2t_infos["test"] = a2t_test_infos
    a2t_infos_df = create_res_df_from_dict(a2t_infos, TaskType.A2T)
    a2t_infos_df.to_csv(args.res_a2t_file_path)
    verbose_results(a2t_infos_df, TaskType.A2T, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_input_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=64)

    parser.add_argument("--task_type", type=str, default=TaskType.T2A.value)
    parser.add_argument("--absa_tuple", type=str, default=AbsaTupleType.ACOP.value)
    parser.add_argument("--annotations_text_type", type=str, 
        default=AnnotationsTextFormerType.GAS_EXTRACTION_STYLE.value)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--annotations_column", type=str, default="annotations")
    parser.add_argument("--annotations_raw_format", type=str, default="acpo",
        help="a: aspect term, c: aspect category, p: sentiment polarity, o: opinion term"
    )
    parser.add_argument("--annotation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(annotation_flag=True)

    parser.add_argument("--train_path", type=str, default="")
    parser.add_argument("--eval_path", type=str, default="")
    parser.add_argument("--test_path", type=str, default="")
    parser.add_argument("--exp_dir", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--train_labeled_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--n_labeled_epochs", type=int, default=10)
    parser.add_argument("--n_unlabeled_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_t2a_model_path", type=str, default="")
    parser.add_argument("--save_a2t_model_path", type=str, default="")
    parser.add_argument("--save_eval_results", action=argparse.BooleanOptionalAction)
    parser.set_defaults(save_eval_results=True)

    parser.add_argument("--lower_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lower_flag=True)
    parser.add_argument("--delete_stopwords_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_stopwords_flag=False)
    parser.add_argument("--delete_punctuation_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_punctuation_flag=False)
    parser.add_argument("--delete_non_ascii_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(delete_non_ascii_flag=True)
    parser.add_argument("--first_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(first_line_flag=False)
    parser.add_argument("--last_line_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(last_line_flag=False)
    parser.add_argument("--stem_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(stem_flag=False)
    parser.add_argument("--lemmatize_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(lemmatize_flag=False)

    args = parser.parse_args()
    main(args)
