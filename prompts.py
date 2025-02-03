# Ben Kabongo
# Feb 2025

# Absa: Prompts

from typing import Any, Tuple
from enums import TaskType


class Prompter(object):

    def __init__(self, args: Any):
        self.args = args

    def get_prompt(
        self, 
        task_type: TaskType=None,
        text: str=None, 
        annotations: Tuple[str]=None
    ) -> str:
        if task_type is None:
            task_type = self.args.task_type
        
        prefix = self.get_prefix(task_type)
        sample = text if task_type is TaskType.T2A else annotations
        prompt = f"{prefix}{sample}"
        return prompt
        
    def get_prefix(self, task_type: TaskType) -> str:
        if task_type is TaskType.T2A:
            return "translate from text to absa tuples: "
        else:
            return "translate from absa tuples to text: "

    def get_text(self, task_type: TaskType, prompt: str) -> str:
        prefix = self.get_prefix(task_type)
        text = prompt[len(prefix):]
        return text


if __name__ == "__main__":
    from enums import AbsaTupleType
    class Args:
        pass

    def test():
        args = Args()
        args.task_type = TaskType.T2A
        args.absa_tuple = AbsaTupleType.ACOP
        text = "The battery life of this laptop is very good."
        prompter = Prompter(args)
        prompt = prompter.get_prompt(text=text)
        print(prompt)

        args.task_type = TaskType.A2T
        args.absa_tuple = AbsaTupleType.ACOP
        annotations = ("battery life", "quality", "good", "positive")
        annotations = "(" + ", ".join(annotations) + ")"
        prompt = prompter.get_prompt(annotations=annotations)
        print(prompt)

    test()