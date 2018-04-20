import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from konlpy.tag import Mecab
from copy import deepcopy
tagger = Mecab()
flatten = lambda l: [item for sublist in l for item in sublist]
from data_utils import *
from model import SDEN
import pickle

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

class ContextNLU:
    def __init__(self):
        self.word2index = pickle.load(open(THIS_PATH+'/vocab.pkl','rb'))
        slot2index = pickle.load(open(THIS_PATH+'/slot.pkl','rb'))
        intent2index = pickle.load(open(THIS_PATH+'/intent.pkl','rb'))
        self.index2intent = {v:k for k,v in intent2index.items()}
        self.index2slot = {v:k for k,v in slot2index.items()}
        self.model = SDEN(len(self.word2index),100,64,len(slot2index),len(intent2index))
        self.model.load_state_dict(torch.load(THIS_PATH+'/sden.pkl'))
        self.model.eval()
        self.history=[Variable(torch.LongTensor([2])).view(1,-1)]
    def reset(self):
        self.history=[Variable(torch.LongTensor([2])).view(1,-1)]
    
    def predict(self,current):
        current = tagger.morphs(current)
        current = prepare_sequence(current,self.word2index).view(1,-1)
        history = pad_to_history(self.history,self.word2index)
        s,i = self.model(history,current)
        slot_p = s.max(1)[1]
        intent_p = i.max(1)[1]
        slot = [self.index2slot[s] for s in slot_p.data.tolist()]
        intent = self.index2intent[intent_p.data[0]]
        
        if len(self.history)==[Variable(torch.LongTensor([2])).view(1,-1)]:
            self.history.pop()
        self.history.append(current)
        
        return slot, intent
                                   
                                   