# from gincov import GINConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, k_hop_subgraph
from bondedgeconstruction import smiles_to_data, collate_with_circle_index
import copy
import random
# random.seed(42)
def set_seed(seed_value=42):
    random.seed(seed_value)
    # np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MS_BACL(nn.Module):
    def __init__(self, num_features_xd=93,dropout=0.5,aug_ratio=0.4):
        super(MS_BACL, self).__init__()

        self.W_rnn = nn.GRU(bidirectional=True, num_layers=1, input_size=100, hidden_size=100)

        self.fc = nn.Sequential(
            nn.Linear(200, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.linear = nn.Sequential(
            nn.Linear(200, 512),
            nn.Linear(512, 256)
        )

        self.fc_g = nn.Sequential(
            nn.Linear(num_features_xd*10*2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_g1 = nn.Sequential(
            nn.Linear(43 * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512)
        )
        self.fc_final = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        self.fc_final1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256 * 1, 1)
        )
        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.conv3 = GINConv(nn.Linear(43, 43))
        self.conv4 = GINConv(nn.Linear(43, 43 * 10))
        self.relu = nn.ReLU()
        self.aug_ratio = aug_ratio
        self.linear2=nn.Linear(300,256)

        r_prime=feature_num=embed_dim = 256
        self.max_walk_len = 3
        self.activation = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        ##
        self.W = torch.nn.Parameter(torch.randn(embed_dim, feature_num), requires_grad=True)
        self.Wv = torch.nn.Parameter(torch.randn(r_prime, embed_dim), requires_grad=True)
        self.Ww = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.Wg = torch.nn.Parameter(torch.randn(r_prime, r_prime), requires_grad=True)
        self.linea1=nn.Linear(57,93)
        self.linea2=nn.Linear(43,93)


    def forward(self, data,x,edge_index,batch,a,edge,c):





        x_g = self.relu(self.conv1(x, edge_index))

        x_g = self.relu(self.conv2(x_g, edge_index))

        x_g = torch.cat([gmp(x_g, batch), gap(x_g, batch)], dim=1)
        x_g = self.fc_g(x_g)
        z = self.fc_final((x_g))
        # a=self.linea2(a)
        x_g1 = self.relu(self.conv3(a, edge))

        x_g1 = self.relu(self.conv4(x_g1,  edge))

        x_g1 = torch.cat([gmp(x_g1, c), gap(x_g1, c)], dim=1)
        x_g1 = self.fc_g1(x_g1)
        z1 = self.fc_final1((x_g1))






        return z,x_g,x_g1,z1

    @staticmethod
    def softmax(input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        soft_max_2d = F.softmax(trans_input.contiguous().view(-1, trans_input.size()[-1]), dim=1)
        return soft_max_2d.view(*trans_input.size()).transpose(axis, len(input_size) - 1)







