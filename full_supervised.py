# Ben Kabongo
# Feb 2025

# Absa: Full supervised

from cycle_absa import *


def task_trainer(
    model,
    tokenizer,
    annotations_text_former,
    train_dataloader,
    test_dataloader,
    args
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task_type is TaskType.T2A:
        args.best_t2a_f1_score = 0.0
    else:
        args.best_a2t_meteor_score = 0.0

    infos = {"test": {}, "training": {}}
    train_infos, test_infos = trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        annotations_text_former=annotations_text_former,
        train_dataloader=train_dataloader,
        train_evaluated_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        task_type=args.task_type,
        n_epochs=args.n_epochs,
        args=args
    )
    infos["training"]["train"] = train_infos
    infos["training"]["eval"] = test_infos

    infos_df = create_res_df_from_dict(infos, args.task_type)
    infos_df.to_csv(args.res_file_path)
    verbose_results(infos_df, args.task_type, args)

    return infos


def main(args):
    set_seed(args)

    args.task_type = TaskType(args.task_type)
    args.absa_tuple = AbsaTupleType(args.absa_tuple)
    args.annotations_text_type = AnnotationsTextFormerType(args.annotations_text_type)

    annotations_text_former = AnnotationsTextFormerBase.get_annotations_text_former(args, )
    prompter = Prompter(args)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

    if args.train_flag:
        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.eval_path)
        test_df = pd.read_csv(args.test_path)

        train_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, train_df, args, task_type=args.task_type,
            split_name="Train"
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )
        
        val_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, val_df, args, task_type=args.task_type,
            split_name="Eval"
        )
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )

        test_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, test_df, args, task_type=args.task_type,
            split_name="Test"
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )

    else:
        test_df = pd.read_csv(args.dataset_path)
        test_dataset = T5ABSADataset(
            tokenizer, annotations_text_former, prompter, test_df, args, task_type=args.task_type,
            split_name="Test"
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            pin_memory=True, num_workers=4
        )

        if args.eval_res_file_path == "":
            args.eval_res_file_path = os.path.join(args.dataset_dir, "absa_results.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_file_path = os.path.join(args.exp_dir, "log.txt")
    args.res_file_path = os.path.join(args.exp_dir, "res.csv")
    args.eval_res_file_path = os.path.join(args.exp_dir, "eval_res.csv")

    args.save_model_path = os.path.join(args.exp_dir, "model.pth")
    if args.task_type is TaskType.T2A:
        args.save_t2a_model_path = args.save_model_path
    else:
        args.save_a2t_model_path = args.save_model_path

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    if args.load_model:
        load_model(model, args.save_model_path)
        print(f"Model loaded from {args.save_model_path} ...\n")

    if args.verbose:
        batch = next(iter(test_dataloader))
        input_texts = batch["input_texts"]
        input_ids = batch["input_ids"]
        log_example = f"Input: {input_texts[0]}"
        log_example += f"\nInputs size: {input_ids.shape}"

        if args.train_flag:
            target_texts = batch["target_texts"]
            annotations = batch["annotations"]
            log_example += f"\nTarget: {target_texts[0]}"
            log_example += f"\nAnnotations: {annotations[0]}"

        log = (
            f"Task: {args.task_type}\n" +
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

    if args.train_flag:
        infos = task_trainer(
            model,
            tokenizer,
            annotations_text_former,
            train_dataloader,
            val_dataloader,
            args
        )
    else:
        infos = {"test": {}}

    load_model(model, args.save_model_path)
    model.to(args.device)
    test_infos = evaluate(
        model, tokenizer, annotations_text_former, test_dataloader, args.task_type, args
    )
    infos["test"] = test_infos
    infos_df = create_res_df_from_dict(infos, args.task_type)
    infos_df.to_csv(args.res_file_path)
    verbose_results(infos_df, args.task_type, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, default="t5-small")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-small")
    parser.add_argument("--max_input_length", type=int, default=32)
    parser.add_argument("--max_target_length", type=int, default=32)

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
    parser.add_argument("--train_dataset_path", type=str, default="")
    parser.add_argument("--val_dataset_path", type=str, default="")
    parser.add_argument("--test_dataset_path", type=str, default="")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--train_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(train_flag=True)
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction)
    parser.set_defaults(load_model=False)
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
