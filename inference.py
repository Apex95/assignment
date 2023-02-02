import logging
import sys
from typing import Optional, Any
import torch
from model import FurnitureTrf
from transformers import DistilBertTokenizer, DistilBertModel

logger = logging.getLogger('furniture-trf-logger')


class FurnitureTrfInterface:
    """A small interface for inference which abstracts some details
    """

    SUPPORTED_TOKENIZERS = ['distilbert-base-uncased']
    SUPPORTED_BACKBONES = ['distilbert-base-uncased']

    def __init__(self, device: Any, checkpoint: Optional[str], backbone: torch.nn.Module, tokenizer: Any):
        """Initializes a transformer model and offers high level access

        Args:
            device (Any): the device it runs on
            checkpoint (Optional[str]): path to an existing checkpoint, or None
            backbone (torch.nn.Module): the pretrained backbone
            tokenizer (Any): the tokenizer which should be used
        """
        assert tokenizer in self.SUPPORTED_TOKENIZERS, logging.error(
            f'Tokenizer {tokenizer} is not currently supported; must be in {self.SUPPORTED_TOKENIZERS}')
        assert backbone in self.SUPPORTED_BACKBONES, logging.error(
            f'Backbone {backbone} is not currently supported; must be in {self.SUPPORTED_BACKBONES}')

        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer)
        self.backbone = DistilBertModel.from_pretrained(backbone)

        self.device = torch.device(device)
        self.model = FurnitureTrf(backbone=self.backbone)
        if checkpoint is not None:
            self.model.load_state_dict(torch.load(
                checkpoint, map_location=self.device))

    def get_model(self) -> torch.nn.Module:
        """Returns the nn.Module

        Returns:
            torch.nn.Module: the transformer model
        """
        return self.model

    def run_inference(self, text: str, threshold: float = 0.5) -> dict:
        """Runs an inference step on the given text

        Args:
            text (str): input text
            threshold (float, optional): the threshold value used. Defaults to 0.5.

        Returns:
            dict: includes thresholded output and confidence values
        """
        self.model.eval()

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            truncation=True
        )

        with torch.no_grad():

            # shape: [batch_size, seq_len]
            ids = torch.tensor(inputs['input_ids']).unsqueeze(
                0).to(self.device, dtype=torch.long)
            mask = torch.tensor(inputs['attention_mask']).unsqueeze(
                0).to(self.device, dtype=torch.long)

            outputs = self.model(ids, mask)
            output = torch.sigmoid(outputs)[0].item()

            thresholded_output = output > threshold
            confidence = output if thresholded_output else 1-output

            return {'output': thresholded_output, 'confidence': confidence}
