import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SomeNet(nn.Module):
    '''
        In : (N, sentence_len)
        Out: (N, sentence_len, embd_size)
    '''
    def __init__(self,
                 seq_len,
                 vocab_size,
                 embd_size,
                 n_layers,
                 kernel,
                 out_chs,
                 res_block_count,
                 ans_size):
        
        super(SomeNet, self).__init__()
        self.res_block_count = res_block_count
        # self.embd_size = embd_size

        self.word_embeddings = nn.Embedding(vocab_size, embd_size)
        
        self.dropout = 0.5
        self.hidden_dim = 128
        self.num_layers = 2

        self.lstm = nn.LSTM(embd_size, self.hidden_dim // 2 , n_layers, dropout=self.dropout, bidirectional=True)

        self.output = nn.Linear(self.hidden_dim, ans_size)

        # # nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, ...
        # self.conv_0 = nn.Conv2d(1, out_chs, kernel_size=kernel, padding=(2, 0))
        # self.b_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))
        # self.conv_gate_0 = nn.Conv2d(1, out_chs, kernel_size=kernel, padding=(2, 0))
        # self.c_0 = nn.Parameter(torch.randn(1, out_chs, 1, 1))

        # self.conv = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        # self.conv_gate = nn.ModuleList([nn.Conv2d(out_chs, out_chs, (kernel[0], 1), padding=(2, 0)) for _ in range(n_layers)])
        # self.b = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])
        # self.c = nn.ParameterList([nn.Parameter(torch.randn(1, out_chs, 1, 1)) for _ in range(n_layers)])

        # self.fc = nn.Linear(out_chs*seq_len, ans_size)
    def attention(self, out, state):

        """
        Use attention to compute soft alignment score between each hidden state and the last hidden state (torch.bmm: batch matrix multiplication)
        """

        hidden = state.squeeze(0)
        attn_weights = torch.bmm(out, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(out.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden
        
    def forward(self, X):
        # x: (N, seq_len)
        # print(X.size())

        #Word embeddings
        embedded = self.word_embeddings(X)
        embedded = embedded.permute(1,0,2)

        #Batch size
        batch_size = X.size(0)

        #Initial hidden state
        h0 = Variable(torch.zeros(2*self.num_layers*5, batch_size, self.hidden_dim // 2)).cuda()
        c0 = Variable(torch.zeros(2*self.num_layers*5, batch_size, self.hidden_dim // 2)).cuda()
        
        #print(h0.size(), c0.size())

        #Forward state
        output, (hidden_state, cell_state) = self.lstm(embedded, (h0, c0))

        x = self.output(output[-1])

        return x
