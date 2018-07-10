import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class SDEN(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,slot_size,intent_size,dropout=0.3,pad_idx=0):
        super(SDEN,self).__init__()
        
        self.pad_idx = 0
        self.embed = nn.Embedding(vocab_size,embed_size,padding_idx=self.pad_idx)
        self.bigru_m = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)
        self.bigru_c = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)
        self.context_encoder = nn.Sequential(nn.Linear(hidden_size*4,hidden_size*2),
                                                               nn.Sigmoid())
        self.session_encoder = nn.GRU(hidden_size*2,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.decoder_1 = nn.GRU(embed_size,hidden_size*2,batch_first=True,bidirectional=True)
        self.decoder_2 = nn.LSTM(hidden_size*4,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.intent_linear = nn.Linear(hidden_size*4,intent_size)
        self.slot_linear = nn.Linear(hidden_size*4,slot_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,history,current):
        batch_size = len(history)
        H= [] # encoded history
        for h in history:
            mask = (h!=self.pad_idx)
            length = mask.sum(1).long()
            embeds = self.embed(h)
            embeds = self.dropout(embeds)
            lens, indices = torch.sort(length, 0, True)
            lens = [l if l>0 else 1 for l in lens.tolist()] # all zero-input
            packed_h = pack(embeds[indices], lens, batch_first=True)
            outputs, hidden = self.bigru_m(packed_h)
            _, _indices = torch.sort(indices, 0)
            hidden = torch.cat([hh for hh in hidden],-1)
            hidden = hidden[_indices].unsqueeze(0)
            H.append(hidden)
        
        M = torch.cat(H) # B,T_C,2H
        M = self.dropout(M)
        
        embeds = self.embed(current)
        embeds = self.dropout(embeds)
        mask = (current!=self.pad_idx)
        length = mask.sum(1).long()
        lens, indices = torch.sort(length, 0, True)
        packed_h = pack(embeds[indices], lens.tolist(), batch_first=True)
        outputs, hidden = self.bigru_c(packed_h)
        _, _indices = torch.sort(indices, 0)
        hidden = torch.cat([hh for hh in hidden],-1)
        C = hidden[_indices].unsqueeze(1) # B,1,2H
        C = self.dropout(C)
        
        C = C.repeat(1,M.size(1),1) 
        CONCAT = torch.cat([M,C],-1) # B,T_c,4H
        
        G = self.context_encoder(CONCAT)
        
        _,H = self.session_encoder(G) # 2,B,2H
        weight = next(self.parameters())
        cell_state = weight.new_zeros(H.size())
        O_1,_ = self.decoder_1(embeds)
        O_1 = self.dropout(O_1)
        
        O_2,(S_2,_) = self.decoder_2(O_1,(H,cell_state))
        O_2 = self.dropout(O_2)
        S = torch.cat([s for s in S_2],1)
        
        intent_prob = self.intent_linear(S)
        slot_prob = self.slot_linear(O_2.contiguous().view(O_2.size(0)*O_2.size(1),-1))
        
        return slot_prob, intent_prob