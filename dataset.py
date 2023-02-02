import logging
import json
import random
import os
from typing import Tuple, List, Any
from torch.utils.data import Dataset
import torch
import string

logger = logging.getLogger('furniture-trf-logger')

class TextDataset(Dataset):
    """A customized dataset for training the furniture transformer (derived from PyTorch's dataset)
    """

    def __init__(self, data: Tuple[List[str], List[int]], tokenizer: Any, max_len: int):
        """Creates a PyTorch-compatible dataset given the data

        Args:
            data (Tuple): the (texts, labels) data in a single tuple
            tokenizer (Any): the tokenizer model that will be used
            max_len (int): the maximum length of the model
        """
        self.text = data[0]
        self.label = data[1]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.text)

    def random_trim(self, text: str) -> str:
        """Randomly takes the part before a `-` (dash) and skips the rest

        Args:
            text (str): the input text

        Returns:
            str: the possibly modified text
        """
        if random.uniform(0, 1) < 0.3:
            text = text.split('-')[0]

        return text

    def random_noise(self, text: str, label: int) -> Tuple[str, int]:
        """Replaces the text with a random "noisy" ASCII sequence

        Args:
            text (str): the input text
            label (int): the input label

        Returns:
            Tuple(str, int): a possibly modified text & label
        """
        if random.uniform(0, 1) < 0.04:
            text = ''.join(random.choices(string.printable, k = random.randint(1, 10)))
            label = 0

        return text, label

    def random_delete(self, text: str, label: int) -> Tuple[str, int]:
        """ Randomly deletes elements from a sequence

        Args:
            text (str): the input text
            label (int): the input label

        Returns:
            Tuple(str, int): a possibly modified text & label
        """
        if random.uniform(0, 1) > .3:
            return text, label

        text = text.split(' ')
        num_elems = random.randint(1, min(3, len(text)))

        for i in range(num_elems):
            rand_idx = random.randrange(len(text))
            text.pop(rand_idx)
        
        if len(text) <= 1:
            label = 0
        text = ' '.join(text)

        return text, label


    def __getitem__(self, idx: int) -> dict:

        text = self.text[idx]
        label = self.label[idx]

        text = self.random_trim(text)
        text, label = self.random_delete(text, label)
        text, label = self.random_noise(text, label)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='longest',
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float)
        }


def load_dataset(json_path: str, tokenizer: Any, max_len: int, split_percent: float = 0.8) -> Tuple[TextDataset, TextDataset, float]:
    """Loads a JSON dataset and returns a splitted version (training/validation)

    Args:
        json_path (str): path to the JSON-formatted datasets
        tokenizer (Any): the tokenizer which is to be used
        max_len (int): maximum seq len for a sample
        split_percent (float, optional): percent of samples used for training. Defaults to 0.8)

    Returns:
        Tuple(TextDataset, TextDataset, float): (train_dataset, val_dataset, neg_to_pos_ratio)
    """
    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []

    pos_samples = 0
    neg_samples = 0
    negative_words_pool = []

    for json_file in os.listdir(json_path):
        with open(f'{json_path}/{json_file}', 'r', encoding='utf-8') as f:
            json_content = json.load(f)

            for seq, label in json_content.items():

                crt_seq = seq.replace('...', '')

                if random.uniform(0, 1) < split_percent:
                    train_inputs.append(crt_seq)
                    train_targets.append(1 if label else 0)

                    if label:
                        pos_samples += 1
                    else:
                        neg_samples += 1
                        negative_words_pool += crt_seq.split(' ')
                else:
                    val_inputs.append(crt_seq)
                    val_targets.append(1 if label else 0)

    for i in range(int(neg_samples * 0.3)):
        random_seq_len = random.randint(1, 30)
        neg_seq = random.choices(negative_words_pool, k=random_seq_len)

        neg_seq = ' '.join(neg_seq)
        train_inputs.append(neg_seq)
        neg_samples += 1
        train_targets.append(0)

    neg_pos_ratio = neg_samples / pos_samples
    logger.info(f'Neg/Pos Ratio: {neg_pos_ratio}')
    logger.info(f'Train Samples: {len(train_inputs)} | Val Samples: {len(val_inputs)}')

    return TextDataset((train_inputs, train_targets), tokenizer, max_len), \
           TextDataset((val_inputs, val_targets), tokenizer, max_len), \
           neg_pos_ratio
                



