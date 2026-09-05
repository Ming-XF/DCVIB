"""模型包。"""

from .cnn import CNN
from .gnn import GCN
from .rnn import RNN
# CNN / GNN / RNN 版 VIB/SVIB/CEB/FGIB/NIB/DVCCA 与 MLP 版同名，需从 model.cnn / model.gnn / model.rnn 导入
from .mlp import (CEB, CEBEnergy, CEBTied, DVCCA, FGIB, MLP, NCMLearn,
                  NCMOrtho, NIB, OPB, OPBFreeScale, SVIB, VIB)
