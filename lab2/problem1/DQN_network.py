import torch.nn as nn

class Network(nn.Module):

    def __init__(self, input_sz, output_sz, hidden_sz=8):
        super().__init__()
        self.layer1 = nn.Linear(input_sz, hidden_sz)
        self.layer1_act = nn.ReLU()
        self.layer2 = nn.Linear(hidden_sz, output_sz)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer1_act(x)
        x = self.layer2(x)
        return x
