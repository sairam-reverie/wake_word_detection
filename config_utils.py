from pathlib import Path

cur_dir = Path.cwd()
data_dir = cur_dir.parent / "data"
audio_dir = data_dir / "audio_data"
wake_word_array_dir = audio_dir / "wake_word_data_array"
trn_dir = wake_word_array_dir / "trn_set"
val_dir = wake_word_array_dir / "val_set"
checkpoint_dir = cur_dir / "checkpoints"

model_params = dict(input_dim=20,
                    num_classes=2,
                    num_heads=4,
                    ffn_dim=128,
                    num_layers=4,
                    depthwise_conv_kernel_size=31)

train_params = dict(trn_batch_size=4,
                  val_batch_size=637,
                  epochs=5)                

optim_params = dict(lr=1e-3)

path_params = dict(trn_dir=trn_dir,
                   val_dir=val_dir,
                   checkpoint_dir=checkpoint_dir)

log_params = dict(check_every=50,
                  validate_every=100,
                  save=True)


config = dict(model_params=model_params,
              train_params=train_params,
              optim_params=optim_params,
              path_params=path_params,
              log_params=log_params)
                