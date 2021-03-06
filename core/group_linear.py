import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, use_bias: bool = False,
                 use_shuffle: bool = False):
        '''

        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        '''
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            print_error_message(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            print_error_message(err_msg)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        if use_bias:
            # add 1 in order to make it broadcastable across batch dimension
            self.bias = nn.Parameter(torch.Tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.n_groups = n_groups
        self.use_bias = use_bias
        self.shuffle = use_shuffle
        self.feature_shuffle = True if use_shuffle else False

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)
        if self.use_bias:
            nn.init.constant_(self.bias.data, 0)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # add bias
        if self.use_bias:
            x = torch.add(x, self.bias)

        if self.feature_shuffle:
            # [g x B x M/g] --> [B x M/g x g]
            x = x.permute(1, 2, 0)
            # [B x M/g x g] --> [B x g x M/g]
            x = x.contiguous().view(bsz, self.n_groups, -1)
        else:
            # [g x B x M/g] --> [B x g x M/g]
            x = x.transpose(0, 1)  # transpose so that batch is first

        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N]
        :return:
        '''
        if x.dim() == 2:
            x = self.process_input_bmm(x)
        elif x.dim() == 3:
            T, B, N = x.size()
            x = x.contiguous().view(B * T, -1)
            x = self.process_input_bmm(x)
            x = x.contiguous().view(T, B, -1)
        else:
            raise NotImplementedError
        return x

    def __repr__(self):
        s = '{name}(in_features={in_features}, out_features={out_features}, num_groups={n_groups}'
        if self.use_bias:
            s += ', bias={use_bias}'
        if self.shuffle:
            s += ', shuffle={shuffle}'

        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def compute_macs_params(self):
        '''
            # of operations in group linear transformation (GLT) are given as:
            Let N and M be dimensions of the input and the output tensor
            Both input and output are split into G groups, so that each input and output group has dimension of N/G and M/G
            Each input group of dimension N/G is mapped to each output group of dimension M/G using a matrix with dimensions [N/G x M/G].
            This mapping involves NM/G^2 additions and NM/G^2 multiplications.
            Since, there are G such groups, we will have total of NM/G addiations and NM/G multipplications.
            Or in simple words, total multiplication-additions (MACs) would be NM/G and FLOPs would be 2NM/G.

            Relationship with # of parameters:
            We have G matrices, each of dimension [N/G x M/G]. The number of parameters in each matrix is NM/G^2.
            Therefore, the total number of parameters in GLT is NM/G.

            MACs = parameters
        '''
        n_mul_wt = self.weights.numel()
        n_add_bias = self.bias.numel() if self.use_bias else 0
        macs = n_mul_wt + n_add_bias
        n_params = n_mul_wt + n_add_bias

        return {
            'name': self.__class__.__name__,
            'macs': macs,
            'params': n_params
        }


if __name__ == '__main__':
    test_data = torch.randn(1,16,256)
    GL = GroupLinear(256,512,use_shuffle=True)
    print(GL)
    print(GL.compute_macs_params())
    print(GL(test_data).size())