from tools.data_utils import get_data_loader
from tools.metrics import get_metrics,get_preds,get_loss
from tools.models import SimpleConformer
from config_utils import config
from pathlib import Path
import torch
from torch.optim import SGD
import wandb
import yaml

project_name = "Wake Word Experiments"  
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Training on {device}")

cur_dir = Path.cwd()
data_dir = cur_dir.parent / "data"
audio_dir = data_dir / "audio_data"
wake_word_array_dir = audio_dir / "wake_word_data_array"
trn_dir = wake_word_array_dir / "trn_set"
val_dir = wake_word_array_dir / "val_set"
rough_dir = cur_dir / "rough"

def train(config): 
    run = wandb.init(project=project_name, entity='sairam_reverie', config = config)
    model_params = config["model_params"]
    optim_params = config["optim_params"]
    epochs = config['train_params']['epochs']
    trn_dir = config['path_params']['trn_dir']
    trn_batch_size = config['train_params']['trn_batch_size']
    val_dir = config['path_params']['val_dir']
    val_batch_size = config['train_params']['val_batch_size']
    pos_labels = config['train_params']['pos_labels']
    check_every = config["log_params"]["check_every"]
    validate_every = config["log_params"]["validate_every"]
    save = config["log_params"]["save"]
    run_dir = Path(wandb.run.dir)
    config_file = run_dir / "train_config.yaml"
    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    checkpoint_dir.mkdir(exist_ok=True,parents=True)
    results_dir.mkdir(exist_ok=True,parents=True)

    yaml_str = yaml.dump(config, default_flow_style=None)
    with open(config_file, 'w+') as file:
        file.write(yaml_str)

    trn_loader = get_data_loader(trn_dir,pos_labels,trn_batch_size)#,balanced=True
    val_loader = get_data_loader(val_dir,pos_labels,val_batch_size)

    model = SimpleConformer(**model_params).to(device)
    optimizer = SGD(model.parameters(), **optim_params)

    iteration = 0
    for epoch in range(1,epochs+1):
        print (f'\nRunning epoch {epoch }')
        for batch in trn_loader:
            model.train()
            
            inputs,labels,lengths,names = batch
            scores = model(batch)
            loss = get_loss(scores,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = get_preds(scores,labels)
            
            labels = labels.tolist()
            iteration += 1

            if iteration%check_every == 0:
                accuracy,f1_score,tn,fp,fn,tp = get_metrics(preds,labels)
                trn_metrics = dict(trn_loss=round(loss.item(),2),
                                   trn_accuracy=accuracy,
                                   trn_f1_score=f1_score,
                                   trn_tn=tn,
                                   trn_fp=fp,
                                   trn_fn=fn,
                                   trn_tp=tp)
                wandb.log(data=trn_metrics,step=iteration)
                print (f"Step {iteration} : Training metrics {trn_metrics}")


        val_metrics,names,labels,preds = valid_step(model,val_loader)
        wandb.log(data=val_metrics,step=iteration)
        print (f"Step {iteration} : Validation metrics {val_metrics}")
        file_name = results_dir / f"Epoch {epoch} Validation Set.csv"
        with open(file_name,"w+") as f:
            f.write(f"File_name,Label,Prediction\n")
            for (name,label,pred) in zip(names,labels,preds):
                f.write(f"{name},{label},{pred}\n")


        if save:
            chkpt_path = checkpoint_dir /  f'model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict(),'loss': loss,}, chkpt_path)

def valid_step(model,val_loader):
    model.eval()
    labels = []
    val_loss = 0
    all_preds = []
    all_labels = []
    all_names = []

    #Special case where entire validation dataset can fit in memory
    for i,batch in enumerate(val_loader):
        with torch.inference_mode():
            inputs,labels,lengths,names = batch
            scores = model(batch)
            preds = get_preds(scores)
            loss = round(get_loss(scores,labels).item(),2)
            val_loss += loss
            all_preds += preds
            all_labels += labels.tolist()
            all_names += names
    
    val_loss /= (i+1)
    accuracy,f1_score,tn,fp,fn,tp = get_metrics(all_preds,all_labels)
    accuracy,f1_score,tn,fp,fn,tp = round(accuracy,2),round(f1_score,2),round(tn,2),round(fp,2),round(fn,2),round(tp,2)
    val_metrics = dict(VAL_LOSS=round(val_loss,2),
                        VAL_ACCURACY=accuracy,
                        VAL_F1_SCORE=f1_score,
                        VAL_TN=tn,
                        VAL_FP=fp,
                        VAL_FN=fn,
                        VAL_TP=tp) 

    return val_metrics,all_names,all_preds,all_labels

if __name__ == "__main__":
    train(config)
