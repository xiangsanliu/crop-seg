
config = dict(
    model= dict(
        type='UNet',
        model_config = dict(
            n_channels=3,
            n_classes=5
        )
    ),
    train_pipeline = dict(
        train_loader = dict(batch_size = 12,num_workers = 12,drop_last = True,pin_memory=True,shuffle=True),
        val_loader = dict(batch_size = 12,num_workers = 12,drop_last = True,pin_memory=False,shuffle=False),

        dataset = dict(type="PNG_Dataset",
                    csv_file=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/train.csv',
                    image_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/image',
                    mask_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/label'),

        transforms = [
            dict(type="RandomCrop",p=1,output_size=(512,512)),
            dict(type="RandomHorizontalFlip",p=0.5),
            dict(type="RandomVerticalFlip",p=0.5),
            # dict(type="ColorJitter",brightness=0.08,contrast=0.08,saturation=0.08,hue=0.08),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],
    ),
    train_config=dict(
        device='cuda',
        lr=1e-3,
        epoches=100,
        restore=False,
        model_type='SegFormer',
        n_classes=5,
        mode='train'
    ),
    lr_scheduler = dict(step_size=10, gamma=0.5, last_epoch=-1)
)