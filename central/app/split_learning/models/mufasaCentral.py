import torch
import torch.nn as nn


class MUFASA(nn.Module):
    def __init__(self):
        super(MUFASA, self).__init__()
        #! # layers for continuous features branch
        # self.input = nn.Linear(contFeatureLen, 128)
        self.inputNorm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4)
        self.lRelu = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.4)
        # ? Addition Layer
        self.nextLayerNorm = nn.LayerNorm(128)
        self.conv1 = nn.Linear(128, 512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.next2LayerNorm = nn.LayerNorm(512)
        self.conv2 = nn.Linear(512, 128)
        # ? Addition Layer

        #! # layers for categorical features branch
        # self.cat_1_input = nn.Linear(categLen, 128)
        self.cat_2_layerNorm = nn.LayerNorm(128)
        self.cat_3_self_attention = nn.MultiheadAttention(128, 4)
        self.cat_4_conv1 = nn.Linear(256, 256)
        self.cat_5_relu = nn.ReLU()
        self.cat_dropout = nn.Dropout(0.4)
        # ? addition
        self.cat_branch_layerNorm = nn.LayerNorm(256)
        self.cat_branch_conv2 = nn.Linear(256, 384)
        self.cat_branch_relu = nn.ReLU()
        self.cat_branch_dropout = nn.Dropout(0.3)

        #! # layers for clinical features branch
        # self.cli_1_input = nn.Linear(clinicNotesLen, 128)
        self.cli_2_layerNorm = nn.LayerNorm(128)
        self.cli_3_selfAtt = nn.MultiheadAttention(128, 4)
        # ? addition
        self.cli_4_layerNorm = nn.LayerNorm(128)
        self.cli_5_conv1 = nn.Linear(128, 512)
        self.cli_6_relu = nn.ReLU()
        self.cli_6_1_dropout = nn.Dropout(0.4)
        self.cli_7_layerNorm = nn.LayerNorm(512)
        self.cli_8_conv2 = nn.Linear(512, 128)
        # ? addition
        # ? Fuse concatenation between ret1 from categorical and current
        self.cli_9_layerNorm = nn.LayerNorm(384)
        self.cli_10_conv3l = nn.Linear(384, 1536)
        self.cli_11_conv3r = nn.Linear(384, 384)
        self.cli_12_relu = nn.ReLU()
        self.cli_12_1_dropout = nn.Dropout(0.3)
        self.cli_13_conv4 = nn.Linear(1536, 384)
        # ? Fuse addition between concat output, current, right branch conv,
        # ? continuous branch, ret2 from categorical branch

        #! Final Output from addition
        # self.out = nn.Linear(384, 128)

    def forward(self, contIn, cateIn, clinIn):
        # TODO contIn, cateIn, clinIn = sample
        # print(contIn.shape, cateIn.shape, clinIn.shape)
        contOutput = self.continuousFeaturesForward(contIn)
        ret1, ret2 = self.categoricalFeaturesForward(cateIn)
        res = self.clinicalFeaturesForward(clinIn, ret1, ret2, contOutput)
        # res = self.out(res)
        return res

    def categoricalFeaturesForward(self, inp):
        # xa = self.cat_1_input(inp)
        x = inp.clone()
        x = self.cat_2_layerNorm(x)
        x = self.cat_3_self_attention(x, x, x, need_weights=False)
        xb = torch.concatenate([x[0], inp], dim=1)
        x = self.cat_4_conv1(xb)
        x = self.cat_dropout(self.cat_5_relu(x))

        xBran = self.cat_branch_layerNorm(xb)
        ret1 = torch.add(xBran, x)
        xBran = self.cat_branch_conv2(xBran)
        ret2 = self.cat_branch_dropout(self.cat_branch_relu(xBran))
        return ret1, ret2

    def clinicalFeaturesForward(self, inp, ret1, ret2, contFeat):
        # xa = self.cli_1_input(inp)
        x = inp.clone()
        x = self.cli_2_layerNorm(x)
        x = self.cli_3_selfAtt(x, x, x, need_weights=False)
        # raise NotImplementedError("Change the skip strategy")
        x = torch.add(x[0], inp)
        xb = self.cli_4_layerNorm(x)
        x = self.cli_5_conv1(xb)
        x = self.cli_6_1_dropout(self.cli_6_relu(x))
        x = self.cli_7_layerNorm(x)
        x = self.cli_8_conv2(x)
        x = torch.add(x, xb)
        xc = torch.concatenate([x, ret1], dim=1)
        x = self.cli_9_layerNorm(xc)
        xdl = self.cli_10_conv3l(x)
        xdr = self.cli_11_conv3r(x)
        x = self.cli_12_1_dropout(self.cli_12_relu(xdl))
        x = self.cli_13_conv4(x)
        # print(x.shape, xc.shape, contFeat.shape, xdr.shape, ret2.shape)
        x = x + xc + nn.functional.pad(contFeat, (0, 256), value=0) + xdr + ret2
        return x

    def continuousFeaturesForward(self, inp):
        # x = self.input(inp)
        sav0 = inp.clone()
        x = self.inputNorm(inp)
        x = self.attention(x, x, x, need_weights=False)
        x = self.dropout1(self.lRelu(x[0]))
        addOutput = torch.add(x, sav0)
        x = self.nextLayerNorm(addOutput)
        x = self.conv1(x)
        x = self.dropout2(self.relu(x))
        x = self.next2LayerNorm(x)
        x = self.conv2(x)
        output = torch.add(x, addOutput)
        return output


class InputEncoder(nn.Module):
    def __init__(self, contLen, categLen, cliLen):
        super(InputEncoder, self).__init__()
        self.contInput = nn.Linear(contLen, 128)
        self.categInput = nn.Linear(categLen, 128)
        self.cliInput = nn.Linear(cliLen, 128)

    def forward(self, cont, categ, cli):
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
