# Model selection part
The selection phase of the training included separate sets of code for training Unet, Attention Unet and Swin UNETR [models](https://github.com/nazarb/2025_levees_DL/tree/main/Model_selection)
- [Unet](https://github.com/nazarb/2025_levees_DL/blob/main/Model_selection/MONAI_UNET_aug.ipynb)
- [Attention Unet](https://github.com/nazarb/2025_levees_DL/blob/main/Model_selection/MONAI_Att_UNET_aug.ipynb)
- [Swin UNETR](https://github.com/nazarb/2025_levees_DL/blob/main/Model_selection/MONAI_SWIN_UNETR_aug.ipynb)


## Unet model
- in_channels=53, # Only used in the model selection phase
- out_channels=1,
- use_checkpoint=True,
- channels=(16, 32, 64, 128, 256),
- strides=(2,2,2,2),
- kernel_size=3,
- up_kernel_size=3,
- dropout=0.0
    
## Attention Unet model
- spatial_dims=2,
- in_channels=53, # Used in the model selection phase and in the final phase
- out_channels=1,
- channels=(16, 32, 64, 128, 256),
- strides=(2,2,2,2),
- kernel_size=3,
- up_kernel_size=3,
- dropout=0.0
      
## Swin UNETR model
- img_size=(96, 96),
- in_channels=53, # Used in the model selection phase and in the final phase
- out_channels=1, 
- use_checkpoint=True,
- feature_size=48,
- depths=(3, 9, 18, 3),
- num_heads=(4, 8, 16, 32),
- drop_rate=0.1, 
- attn_drop_rate=0.1,
- dropout_path_rate=0.2,
- spatial_dims=2
