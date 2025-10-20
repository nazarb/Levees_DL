## The final model

The final version of the [Swin UNETR](https://github.com/nazarb/2025_levees_DL/tree/main/Swin_UNETR) model consist of two part:
1. [Train the model](https://github.com/nazarb/2025_levees_DL/blob/main/Swin_UNETR/Train.ipynb)
2. [Predict the levees](https://github.com/nazarb/2025_levees_DL/blob/main/Swin_UNETR/Detect.ipynb)

### Swin UNETR model
- img_size=(96, 96),
- in_channels=48,
- out_channels=1,  # Use the passed `num_classes`
- use_checkpoint=True,
- feature_size=48,
- depths=(3, 9, 18, 3),
- num_heads=(4, 8, 16, 32),
- drop_rate=0.1,  # Added dropout
- attn_drop_rate=0.1,
- dropout_path_rate=0.2,
- spatial_dims=2

### Train parametres


### Train
The training parametres of the model :
- Dataset: dataset E (see Publication)
- Image size: 96x96x48
- Epochs: 500
- Early Stopping: False
- Optimizer: [RAdam](https://docs.pytorch.org/docs/stable/generated/torch.optim.RAdam.html)
- Loss function: Dice Loss
- Initial learning Rate:2e-4
- Weight Decay: 5e-6
- Scheduler for Learning Rate: True
```
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',      # Reduce learning rate (lr) when metric stops improving
        factor=0.5,      # Reduce lr by 50%
        patience=10,     # Number of epochs with no improvement after which lr will be reduced
        verbose=True,    # Print a message when lr changes
        min_lr=1e-6      # Minimum learning rate
    )
    scaler = torch.cuda.amp.GradScaler()
```
