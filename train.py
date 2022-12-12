from tools.data_utils import get_data_loader
from tools.metrics import get_metrics,get_preds,get_loss
from tools.models import SimpleConformer
from config_utils import config
from pathlib import Path
import torch
from torch.optim import SGD
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Training on {device}")

cur_dir = Path.cwd()
data_dir = cur_dir.parent / "data"
audio_dir = data_dir / "audio_data"
wake_word_array_dir = audio_dir / "wake_word_data_array"
trn_dir = wake_word_array_dir / "trn_set"
val_dir = wake_word_array_dir / "val_set"
checkpoint_dir = cur_dir / "checkpoints"
rough_dir = cur_dir / "rough"

def train(config): 
    model_params = config["model_params"]
    optim_params = config["optim_params"]
    epochs = config['train_params']['epochs']
    trn_dir = config['path_params']['trn_dir']
    trn_batch_size = config['train_params']['trn_batch_size']
    val_dir = config['path_params']['val_dir']
    val_batch_size = config['train_params']['val_batch_size']
    check_every = config["log_params"]["check_every"]
    validate_every = config["log_params"]["validate_every"]
    save = config["log_params"]["save"]

    trn_loader = get_data_loader(trn_dir,trn_batch_size)
    val_loader = get_data_loader(val_dir,val_batch_size)

    model = SimpleConformer(**model_params).to(device)
    optimizer = SGD(model.parameters(), **optim_params)

    iteration = 0
    for epoch in range(1,epochs+1):
        print (f'\nRunning epoch {epoch }')
        all_files = []
        all_labels = []
        all_preds = []
        for batch in trn_loader:
            model.train()
            
            inputs,labels,lengths,names = batch
            scores = model(batch)
            loss = get_loss(scores,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            labels = labels.tolist()
            iteration += 1
            all_files += names
            all_labels += labels
            all_preds += get_preds(scores)

            if iteration % check_every == 0:
                print (f"Step {iteration} : Training Loss {loss}")


        trn_metrics = get_metrics(all_preds,all_labels)
        val_metrics,val_names,val_preds,val_labels= valid_step(model,val_loader,epoch)
        print (f"Epoch {epoch} : Training metrics {trn_metrics}")
        print (f"Epoch {epoch} : Validation metrics {val_metrics}")

                
                
        if save:
            chkpt_path = checkpoint_dir /  f'model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict(),'loss': loss,}, chkpt_path)

def valid_step(model,val_loader,epoch):
    model.eval()
    labels = []

    #Special case where entire validation dataset can fit in memory
    for _,batch in enumerate(val_loader):
        with torch.inference_mode():
            inputs,labels,lengths,names = batch
            scores = model(batch)
            loss = get_loss(scores,labels).item()
            preds = get_preds(scores)
            all_metrics = get_metrics(preds,labels.tolist())
            val_metrics = dict(loss=loss)
            val_metrics = {**val_metrics,**all_metrics}

    file_name = rough_dir / f"Epoch {epoch} Validation Set.csv"
    with open(file_name,"w+") as f:
        f.write(f"File_name,Label,Prediction\n")
        for (name,label,pred) in zip(names,labels,preds):
            f.write(f"{name},{label},{pred}\n")

    return val_metrics,names,preds,labels

if __name__ == "__main__":
    train(config)
