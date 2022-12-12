import torch.nn.functional as F
import torch
from torchmetrics.classification import BinaryAccuracy,BinaryF1Score
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

accuracy_fn = BinaryAccuracy().to(device)
f1_score_fn = BinaryF1Score().to(device)

weight = torch.Tensor([0.1,0.9]).to(device)

def get_preds(scores,threshold=0.5):
    probs = F.softmax(scores,dim=-1)
    pos_probs = probs[:,-1]
    preds = (pos_probs>threshold).long()
    return preds

def get_loss(scores,labels,weight=None):
    return F.cross_entropy(scores,labels,weight)

def get_accuracy(preds,labels):
    return accuracy_fn(preds,labels)

def get_f1_score(preds,labels):
    return f1_score_fn(preds,labels)

def get_metrics(scores,labels):
    preds = get_preds(scores)
    loss = get_loss(scores,labels)
    accuracy = get_accuracy(preds,labels)
    f1_score = get_f1_score(preds,labels)
    return {"loss":loss,"accuracy":accuracy,"f1_score":f1_score}