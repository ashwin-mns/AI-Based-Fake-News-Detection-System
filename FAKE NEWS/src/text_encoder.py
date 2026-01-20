import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', freeze_bert=False):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # return CLS token representation [Batch, HiddenSize]
        return outputs.last_hidden_state[:, 0, :]
