import torch
import transformers
import io
import json
import logging

from dataclasses import dataclass
from typing import Dict, Sequence
from torch.utils.data import Dataset





class DataModule:
    def __init__():
        pass
    
def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


class ChatDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.AutoTokenizer, max_tokens=None):
        super(ChatDataset, self).__init__()
        logging.warning("Loading data...")
        conversations = jload(data_path)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(conversations, tokenizer, max_tokens)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForChatDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class ChatDataModule(DataModule):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path_train: str, max_tokens = None):

        self.train_dataset = ChatDataset(tokenizer=tokenizer, data_path=data_path_train, max_tokens=max_tokens)
        self.data_collator = DataCollatorForChatDataset(tokenizer=tokenizer)


def preprocess(conversations: Sequence[Sequence[dict]], tokenizer: transformers.PreTrainedTokenizer, max_tokens=None) -> Dict:
    """
    Preprocess the data by tokenizing.
    """

    all_input_ids = []
    all_labels = []

    for conv in conversations:
        roles = [msg["role"] for msg in conv]
        messages = [msg["content"] for msg in conv]
        
#         assert roles[0] == "SYSTEM"
        assert roles[0] != "ASSISTANT"
        assert roles[-1] == "ASSISTANT"

        input_messages = []
            
#         input_messages.append(messages[0])

        init = 0
        for role, msg in zip(roles, messages):
            if role == "ASSISTANT":
                input_messages.append(msg)
            elif role == "USER":
                input_messages.append(msg)

        tokenized_input = tokenizer(input_messages, add_special_tokens=False)
#         print(tokenized_input.input_ids[0])

        input_ids = []
        labels = []

        if roles[0] == "SYSTEM":
            input_ids.extend([64790, 64792, 64794, 30910,    13])
            input_ids.extend(tokenized_input.input_ids[0])
            labels.extend([-100]* (len(tokenized_input.input_ids[0]) + 5))
        else:
            input_ids.extend([64790, 64792])
            labels.extend([-100]* 2)

        for role, msg in zip(roles, tokenized_input.input_ids):

            if role == "USER":
                if roles[0] == "SYSTEM":
                    labels.extend([-100]*(len(msg)+5))
                    input_ids.extend([13, 64795, 30910,    13])
                else:
                    labels.extend([-100]*(len(msg)+4))
                    input_ids.extend([64795, 30910,    13])
                input_ids.extend(msg)
                input_ids.extend([64796])
                print("USER",msg)
            
            elif role == "ASSISTANT":
                
                msg += [tokenizer.eos_token_id]
                labels.extend([30910, 13])
                labels.extend(msg)
                input_ids.extend([30910, 13])
                input_ids.extend(msg)
                print("ASSISTANT",msg)

        if max_tokens is None:
            max_tokens = tokenizer.model_max_length

        input_ids = torch.LongTensor(input_ids)[:max_tokens]
        labels = torch.LongTensor(labels)[:max_tokens]

        assert input_ids.shape == labels.shape   

        all_input_ids.append(input_ids)  
        all_labels.append(labels)


    return dict(input_ids=all_input_ids, labels=all_labels)
