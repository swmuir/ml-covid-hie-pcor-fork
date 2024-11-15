import torch.nn as nn
import torch


class InputEncoder(nn.Module):
    def __init__(self, contLen, categLen, cliLen):
        super(InputEncoder, self).__init__()
        self.contInput = nn.Linear(contLen, 128)
        self.categInput = nn.Linear(categLen, 128)
        self.cliInput = nn.Linear(cliLen, 128)

    def forward(self, sample):
        cont, categ, cli = sample
        cont = self.contForward(cont)
        categ = self.categForward(categ)
        cli = self.cliForward(cli)
        return torch.stack([cont, categ, cli])

    def contForward(self, cont):
        return self.contInput(cont)

    def categForward(self, categ):
        return self.categInput(categ)

    def cliForward(self, cli):
        return self.cliInput(cli)


class OutputDecoder(nn.Module):
    def __init__(self, outSize):
        super(OutputDecoder, self).__init__()
        self.out = nn.Linear(384, outSize)
        # self.act = nn.Softmax()

    def forward(self, x):
        return self.out(x)
