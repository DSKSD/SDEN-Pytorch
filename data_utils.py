import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import random
from tqdm import tqdm
flatten = lambda l: [item for sublist in l for item in sublist]


def prepare_dataset(path,built_vocab=None,user_only=False):
    data = open(path,"r",encoding="utf-8").readlines()
    p_data=[]
    history=[["<null>"]]
    for d in data:
        if d=="\n":
            history=[["<null>"]]
            continue
        dd = d.replace("\n","").split("|||")
        if len(dd)==1:
            if user_only:
                pass
            else:
                bot = dd[0].split()
                history.append(bot)
        else:
            user = dd[0].split()
            tag = dd[1].split()
            intent = dd[2]
            temp = deepcopy(history)
            p_data.append([temp,user,tag,intent])
            history.append(user)
    
    if built_vocab is None:
        historys, currents, slots, intents = list(zip(*p_data))
        vocab = list(set(flatten(currents)))
        slot_vocab = list(set(flatten(slots)))
        intent_vocab = list(set(intents))
        
        word2index={"<pad>" : 0, "<unk>" : 1, "<null>" : 2, "<s>" : 3, "</s>" : 4}
        for vo in vocab:
            if word2index.get(vo)==None:
                word2index[vo] = len(word2index)

        slot2index={"<pad>" : 0}
        for vo in slot_vocab:
            if slot2index.get(vo)==None:
                slot2index[vo] = len(slot2index)

        intent2index={}
        for vo in intent_vocab:
            if intent2index.get(vo)==None:
                intent2index[vo] = len(intent2index)
    else:
        word2index, slot2index, intent2index = built_vocab
        
    for t in tqdm(p_data):
        for i,history in enumerate(t[0]):
            t[0][i] = prepare_sequence(history, word2index).view(1, -1)

        t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[2] = prepare_sequence(t[2], slot2index).view(1, -1)
        t[3] = torch.LongTensor([intent2index[t[3]]]).view(1,-1)
            
    if built_vocab is None:
        return p_data, word2index, slot2index, intent2index
    else:
        return p_data

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<unk>"], seq))
    return torch.LongTensor(idxs)

def data_loader(train_data,batch_size,shuffle=False):
    if shuffle: random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch
    
    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch, w_to_ix,s_to_ix): # for bAbI dataset
    history,current,slot,intent = list(zip(*batch))
    max_history = max([len(h) for h in history])
    max_len = max([h.size(1) for h in flatten(history)])
    max_current = max([c.size(1) for c in current])
    max_slot = max([s.size(1) for s in slot])
    
    historys, currents, slots = [], [], []
    for i in range(len(batch)):
        history_p_t = []
        for j in range(len(history[i])):
            if history[i][j].size(1) < max_len:
                history_p_t.append(torch.cat([history[i][j], torch.LongTensor([w_to_ix['<pad>']] * (max_len - history[i][j].size(1))).view(1, -1)], 1))
            else:
                history_p_t.append(history[i][j])

        while len(history_p_t) < max_history:
            history_p_t.append(torch.LongTensor([w_to_ix['<pad>']] * max_len).view(1, -1))

        history_p_t = torch.cat(history_p_t)
        historys.append(history_p_t)

        if current[i].size(1) < max_current:
            currents.append(torch.cat([current[i], torch.LongTensor([w_to_ix['<pad>']] * (max_current - current[i].size(1))).view(1, -1)], 1))
        else:
            currents.append(current[i])

        if slot[i].size(1) < max_slot:
            slots.append(torch.cat([slot[i], torch.LongTensor([s_to_ix['<pad>']] * (max_slot - slot[i].size(1))).view(1, -1)], 1))
        else:
            slots.append(slot[i])

    currents = torch.cat(currents)
    slots = torch.cat(slots)
    intents = torch.cat(intent)
    
    return historys, currents, slots, intents

def pad_to_history(history, x_to_ix): # this is for inference
    
    max_x = max([len(s) for s in history])
    x_p = []
    for i in range(len(history)):
        h = prepare_sequence(history[i],x_to_ix).unsqueeze(0)
        if len(history[i]) < max_x:
            x_p.append(torch.cat([h,torch.LongTensor([x_to_ix['<pad>']] * (max_x - h.size(1))).view(1, -1)], 1))
        else:
            x_p.append(h)
        
    history = torch.cat(x_p)
    return [history]