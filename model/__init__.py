"""模型包。"""

from .cnn import CNN
from .gnn import GCN
# CNN / GNN 版 VIB/SCVIB/DCVIB 与 MLP 版同名，需从 model.cnn / model.gnn 导入
from .mlp import DCVIB, MLP, SCVIB, VIB
