import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from data_utils import *
from model import SDEN
from sklearn_crfsuite import metrics
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluation(model,dev_data):
    model.eval()
    index2slot = {v:k for k,v in model.slot_vocab.items()}
    preds=[]
    labels=[]
    hits=0
    with torch.no_grad():
        for i,batch in enumerate(data_loader(dev_data,32,True)):
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
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
    
    # this is because sklearn_crfsuite.metrics function flatten inputs
    preds = [[y] for y in preds] 
    labels = [[y] for y in labels]
    
    print(metrics.flat_classification_report(
    labels, preds, labels = sorted_labels, digits=3
    ))

def save(model,config):
    checkpoint = {
                'model': model.state_dict(),
                'vocab': model.vocab,
                'slot_vocab' : model.slot_vocab,
                'intent_vocab' : model.intent_vocab,
                'config' : config,
            }
    torch.save(checkpoint,config.save_path)
    print("Model saved!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=5,
                        help='num_epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning_rate')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout')
    parser.add_argument('--embed_size', type=int, default=100,
                        help='embed_size')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='hidden_size')
    parser.add_argument('--save_path', type=str, default='weight/model.pkl',
                        help='save_path')
    
    config = parser.parse_args()
    
    train_data, word2index, slot2index, intent2index = prepare_dataset('data/train.iob')
    dev_data = prepare_dataset('data/dev.iob',(word2index,slot2index,intent2index))
    model = SDEN(len(word2index),config.embed_size,config.hidden_size,\
                 len(slot2index),len(intent2index),word2index['<pad>'])
    model.to(device)
    model.vocab = word2index
    model.slot_vocab = slot2index
    model.intent_vocab = intent2index
    
    slot_loss_function = nn.CrossEntropyLoss(ignore_index=0)
    intent_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=config.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,milestones=[config.epochs//4,config.epochs//2],optimizer=optimizer)
    
    model.train()
    for epoch in range(config.epochs):
        losses=[]
        scheduler.step()
        for i,batch in enumerate(data_loader(train_data,config.batch_size,True)):
            h,c,slot,intent = pad_to_batch(batch,model.vocab,model.slot_vocab)
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
                print("[%d/%d] [%d/%d] mean_loss : %.3f" % \
                      (epoch,config.epochs,i,len(train_data)//config.batch_size,np.mean(losses)))
                losses=[]
                      
    evaluation(model,dev_data)
    save(model,config)