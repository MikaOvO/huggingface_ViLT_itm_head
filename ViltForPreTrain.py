import torch
from torch import nn
from transformers import ViltProcessor, ViltForMaskedLM, ViltForQuestionAnswering, ViltModel

class ViltForPreTrain(nn.Module):
    def __init__(self, path, hidden_size=768):
        super(ViltForPreTrain, self).__init__()
        torch_bin = torch.load(path)
        self.model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm-itm")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dense.weight.data = torch_bin['vilt.pooler.dense.weight'].to(self.device)
        self.dense.bias.data = torch_bin['vilt.pooler.dense.bias'].to(self.device)
        self.activation = nn.Tanh()
        self.fc = nn.Linear(hidden_size, 2)
        self.fc.weight.data = torch_bin['itm_score.fc.weight'].to(self.device)
        self.fc.bias.data = torch_bin['itm_score.fc.bias'].to(self.device)
        self.itm_loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs, head='mlm', labels=None):
        if head == 'itm':
            inputs['output_hidden_states'] = True
        if head == 'mlm':
            inputs['labels'] = labels
        outputs = self.model(**inputs)
        if head == 'itm':
            outputs = outputs['hidden_states'][-1]
            outputs = outputs[:, 0]
            pooled_output = self.dense(outputs)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.fc(pooled_output)
            if labels != None:
                return pooled_output, self.itm_loss_func(pooled_output, labels)
            return pooled_output, None
        if labels != None:
            return outputs['logits'], outputs['loss']
        return outputs['logits'], None

