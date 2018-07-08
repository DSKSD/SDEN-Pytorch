import torch
import torch.nn as nn
import torch.nn.functional as F

class SDEN(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,slot_size,intent_size):
        super(SDEN,self).__init__()
        
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.bigru_m = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)
        self.bigru_c = nn.GRU(embed_size,hidden_size,batch_first=True,bidirectional=True)
        self.context_encoder = nn.Sequential(nn.Linear(hidden_size*4,hidden_size*2),
                                                               nn.Sigmoid())
        self.session_encoder = nn.GRU(hidden_size*2,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.decoder_1 = nn.GRU(embed_size,hidden_size*2,batch_first=True,bidirectional=True)
        self.decoder_2 = nn.LSTM(hidden_size*4,hidden_size*2,batch_first=True,bidirectional=True)
        
        self.intent_linear = nn.Linear(hidden_size*4,intent_size)
        self.slot_linear = nn.Linear(hidden_size*4,slot_size)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self,history,current):
        batch_size = len(history)
        H= [] # encoded history
        for h in history:
            mask = h.eq(0)
            embeds = self.embed(h)
            embeds = self.dropout(embeds)
            outputs, hidden = self.bigru_m(embeds)
            real_hidden = []

            for i, o in enumerate(outputs): # B,T,D
                real_length = mask[i].data.tolist().count(0) 
                real_hidden.append(o[real_length - 1])

            H.append(torch.cat(real_hidden).view(h.size(0), -1).unsqueeze(0))
        
        M = torch.cat(H) # B,T_C,2H
        M = self.dropout(M)
        embeds = self.embed(current)
        embeds = self.dropout(embeds)
        mask = current.eq(0)
        outputs, hidden = self.bigru_c(embeds)
        real_hidden=[]
        for i, o in enumerate(outputs): # B,T,D
            real_length = mask[i].data.tolist().count(0) 
            real_hidden.append(o[real_length - 1])
        C = torch.cat(real_hidden).view(current.size(0),1, -1) # B,1,2H
        C = self.dropout(C)
        
        CONCAT = []
        for i in range(batch_size):
            m = M[i] # T_c,2H
            c = C[i] # 1,2H
            c = c.expand_as(m)
            cat = torch.cat([m,c],1)
            CONCAT.append(cat.unsqueeze(0))
        CONCAT = torch.cat(CONCAT)
        
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