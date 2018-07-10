import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluation(model,dev_data,index2slot,intent2slot):
    model.eval()
    preds=[]
    labels=[]
    hits=0
    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data,32,True)):
            h,c,slot,intent = pad_to_batch(batch,word2index,slot2index)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            slot_p, intent_p = model(h,c)

            preds.extend([index2slot[i] for i in slot_p.max(1)[1].tolist()])
            labels.extend([index2slot[i] for i in slot.view(-1).tolist()])
            hits+=torch.eq(intent_p.max(1)[1],intent.view(-1)).sum().item()


    print(hits/len(dev_data))
    
    sorted_labels = sorted(
    list(set(labels) - {'O','<pad>'}),
    key=lambda name: (name[1:], name[0])
    )
    
    preds = [[y] for y in preds] # this is because sklearn_crfsuite.metrics function flatten inputs
    labels = [[y] for y in labels]
    
    print(metrics.flat_classification_report(
    labels, preds, labels = sorted_labels, digits=3
    ))


if __name__ == "__main__":
    
    train_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob')
    dev_data = prepare_dataset('data/dev.iob',(word2index,slot2index,intent2index))
    index2slot = {v:k for k,v in slot2index.items()}
    index2intent = {v:k for k,v in intent2index.items()}
    EPOCH = 5
    BATCH = 32
    LR = 0.001
    model = SDEN(len(word2index),100,64,len(slot2index),len(intent2index),word2index['<pad>'])
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,milestones=[EPOCH//4,EPOCH//2],optimizer=optimizer)
    
    model.train()
    for epoch in range(EPOCH):
        losses=[]
        scheduler.step()
        for i,batch in enumerate(data_loader(train_data,BATCH,True)):
            h,c,slot,intent = pad_to_batch(batch,word2index,slot2index)
            h = [hh.to(device) for hh in h]
            c = c.to(device)
            slot = slot.to(device)
            intent = intent.to(device)
            model.zero_grad()
            slot_p, intent_p = model(h,c)

            loss_s = slot_loss_function(slot_p,slot.view(-1))
            loss_i = intent_loss_function(intent_p,intent.view(-1))
            loss = loss_s + loss_i
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % (epoch,EPOCH,i,len(train_data)//BATCH,np.mean(losses)))
                losses=[]
                
                
    evaluation(model,dev_data,index2slot,index2intent)