import torch
import torch.nn as nn
from torchaudio.models import Conformer
import torch.nn.functional as F

class SimpleConformer(nn.Module):
    def __init__(self, input_dim,num_classes=2,num_heads=4,ffn_dim=128,num_layers=4,depthwise_conv_kernel_size=31):
        super().__init__()
        self.conformer = Conformer(input_dim=input_dim,
                              num_heads=num_heads,
                              ffn_dim=ffn_dim,
                              num_layers=num_layers,
                              depthwise_conv_kernel_size=depthwise_conv_kernel_size)
        hidden_dim = input_dim//2
        self.feature_extractor = nn.Linear(in_features=input_dim*2, out_features=input_dim//2)
        self.classifier = nn.Linear(hidden_dim,num_classes)
        
    def forward(self,batch):
        samples,labels,lengths = batch
        outputs = self.conformer(samples, lengths)
        output_frames,output_lengths = outputs
        max_output,_ = torch.max(output_frames,1)
        mean_output = torch.mean(output_frames,1)
        output = torch.concat([max_output,mean_output],-1)
        features = F.relu(self.feature_extractor(output))
        scores =  self.classifier(features)
        return scores #[Batch_size,Num_classes]

# input_dim = 20
# batch_size =16
# max_seq_len = 200
# conformer = Conformer(
#     input_dim=input_dim,
#     num_heads=4,
#     ffn_dim=128,
#     num_layers=4,
#     depthwise_conv_kernel_size=31)
# input.shape,lengths.shape
# torch.Size([16, 195, 20]), torch.Size([16])
# lengths = torch.randint(1, max_seq_len, (batch_size,))  # (batch,)
# input = torch.rand(batch_size, int(lengths.max()),input_dim)  # (batch, num_frames, input_dim)
        