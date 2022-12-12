from tools.data_utils import get_data_loader
from tools.metrics import get_metrics,get_preds
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
        print ('\n')
        print (f'Running epoch {epoch }')
        for batch in trn_loader:
            model.train()
            inputs,labels,lengths = batch
            scores = model(batch)
            metrics = get_metrics(scores,labels)
            loss = metrics["loss"]
            loss.backward()
            optimizer.step()
            iteration += 1

            if iteration % check_every == 0:
                print (f"Step {iteration} : Training Loss {metrics['loss']}, Accuracy {metrics['accuracy']}, F1_Score {metrics['f1_score']}")

            if iteration % validate_every == 0:
                val_metrics= valid_step(model,val_loader)
                print (f"Step {iteration} : Validation Loss {val_metrics['val_loss']},Validation Accuracy {val_metrics['val_accuracy']},Validation F1_Score {val_metrics['val_f1_score']}")

                
                
        if save:
            chkpt_path = checkpoint_dir /  f'model_epoch_{epoch}.pt'
            torch.save({'epoch': epoch,'model_state': model.state_dict(),'optimizer_state': optimizer.state_dict(),'loss': loss,}, chkpt_path)

def valid_step(model,val_loader):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_f1_score = 0
    all_labels = []

    for iteration,batch in enumerate(val_loader):
        with torch.inference_mode():
            scores = model(batch)
            inputs,labels,lengths = batch
            metrics = get_metrics(scores,labels)
            preds = get_preds(scores).tolist()
            val_loss += metrics['loss'].item()

            val_accuracy += metrics['accuracy'].item()
            val_f1_score += metrics['f1_score'].item()
    
    val_loss /= (iteration+1)
    val_accuracy /= (iteration+1)
    val_f1_score /= (iteration+1)
        
    val_metrics =  dict(val_loss=val_loss,
                        val_accuracy=val_accuracy,
                        val_f1_score=val_f1_score)
    return val_metrics

if __name__ == "__main__":
    train(config)
