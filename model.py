import torch


class FurnitureTrf(torch.nn.Module):
    """The is_product() transformer for furniture
    """

    def __init__(self, backbone: torch.nn.Module):
        """Creates an instance of the transformer w/ provided backbone

        Args:
            backbone (torch.nn.Module): _description_
        """
        super(FurnitureTrf, self).__init__()
        self.trf = backbone
        self.fc_h1 = torch.nn.Linear(768, 768)
        self.fc_out = torch.nn.Linear(768, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Basic PyTorch forward function

        Args:
            input_ids (torch.Tensor): token ids generated by tokenizer
            attention_mask (torch.Tensor): attention mask from the tokenizer

        Returns:
            torch.Tensor: a [batch_size]-shaped tensor w/ logits
        """
        base_output = self.trf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # extracts last hidden states of BERT which will be needed for classification
        hidden_state = base_output['last_hidden_state']

        # [batch_size, seq_len, hidden_size] -> get first seq item (CLS)
        cls_token = hidden_state[:, 0, :]

        cls_token = self.fc_h1(cls_token)
        cls_token = torch.nn.functional.gelu(cls_token)

        output = self.fc_out(cls_token)

        return output[:, 0]
