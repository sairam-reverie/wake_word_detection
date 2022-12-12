import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score

from torchmetrics.classification import BinaryAccuracy,BinaryF1Score
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

accuracy_fn = BinaryAccuracy().to(device)
f1_score_fn = BinaryF1Score().to(device)

weight = torch.Tensor([0.1,0.9]).to(device)

def get_loss(scores,labels,weight=None):
    return F.cross_entropy(scores,labels,weight)

def get_preds(scores,threshold=0.5):
    probs = F.softmax(scores,dim=-1)
    pos_probs = probs[:,-1]
    preds = (pos_probs>threshold).long().tolist()
    return preds

def get_accuracy(preds,labels):
    return accuracy_score(labels,preds, normalize=True)

def get_f1_score(preds,labels):
    return f1_score(labels,preds)

def get_confusion_matrix(preds,labels):
    conf_matrix = confusion_matrix(labels,preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    return (tn, fp, fn, tp)

def get_metrics(preds,labels):
    accuracy = get_accuracy(preds,labels)
    f1_score = get_f1_score(preds,labels)
    tn, fp, fn, tp = get_confusion_matrix(preds,labels)
    return {"accuracy":accuracy,"f1_score":f1_score,"tn":tn,"fp":fp,"fn":fn,"tp":tp}
