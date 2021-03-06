
config = dict(
    model=dict(
        type='DeepLabV3Plus',
        model_config=dict(
            num_classes=5,
            backbone_config=dict(
                type="resnet50",
                pretrained=True,
                replace_stride_with_dilation=[False, False, 2]
            ),
            head_config=dict(
                in_channels=2048,
                out_channels=256,
                dilation_list=[6, 12, 18]
            )
        )
    ),
    loss=dict(
        type="LabelSmoothing",
        win_size=11,
        num_classes=5
    ),
    train_pipeline=dict(
        train_loader=dict(batch_size=8, num_workers=8,
                          drop_last=True, pin_memory=True, shuffle=True),
        val_loader=dict(batch_size=8, num_workers=8,
                        drop_last=True, pin_memory=False, shuffle=False),

        dataset=dict(type="PNG_Dataset",
                     csv_file=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/image_1.csv',
                     image_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/image',
                     mask_dir=r'/home/xiangjianjian/Projects/spectral-setr/dataset/tianchi/round1/label'),

        transforms=[
            dict(type="RandomCrop", p=1, output_size=(512, 512)),
            dict(type="RandomHorizontalFlip", p=0.5),
            dict(type="RandomVerticalFlip", p=0.5),
            # dict(type="ColorJitter",brightness=0.08,contrast=0.08,saturation=0.08,hue=0.08),
            dict(type="ToTensor",),
            dict(type="Normalize", mean=[0.485, 0.456, 0.406], std=[
                 0.229, 0.224, 0.225], inplace=True),
        ],
    ),
    train_config=dict(
        device='cuda',
        lr=1e-4,
        epoches=100,
        restore=False,
        model_type='DeepLabV3Plus',
        n_classes=5,
        mode='train'
    ),
    lr_scheduler=dict(step_size=10, gamma=0.5, last_epoch=-1)
)
