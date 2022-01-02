dataset_path = "datasets/tianchi/round2_no_overlap"

config = dict(
    model=dict(
        type="DeepLabV3Plus",
        model_config=dict(
            num_classes=5,
            backbone_config=dict(
                type="resnet50",
                pretrained=True,
                replace_stride_with_dilation=[False, False, 2],
                in_channels=3,
            ),
            head_config=dict(
                in_channels=2048, out_channels=256, dilation_list=[6, 12, 18]
            ),
        ),
    ),
    loss=dict(type="LabelSmoothing", win_size=11, num_classes=5),
    train_pipeline=dict(
        dataloader=dict(
            batch_size=8, num_workers=8, drop_last=True, pin_memory=True, shuffle=True
        ),
        dataset=dict(
            type="PNG_Dataset",
            csv_file=f"{dataset_path}/train.csv",
            image_dir=f"{dataset_path}/image",
            mask_dir=f"{dataset_path}/label",
        ),
        transforms=[
            dict(type="RandomHorizontalFlip", p=0.5),
            dict(type="RandomVerticalFlip", p=0.5),
            dict(
                type="ColorJitter",
                brightness=0.08,
                contrast=0.08,
                saturation=0.08,
                hue=0.08,
            ),
            dict(type="ToTensor"),
            dict(
                type="Normalize",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True,
            ),
        ],
    ),
    test_pipeline=dict(
        dataloader=dict(
            batch_size=8, num_workers=8, drop_last=True, pin_memory=False, shuffle=False
        ),
        dataset=dict(
            type="PNG_Dataset",
            csv_file=f"{dataset_path}/test.csv",
            image_dir=f"{dataset_path}/image",
            mask_dir=f"{dataset_path}/label",
        ),
        transforms=[
            dict(type="ToTensor"),
            dict(
                type="Normalize",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                inplace=True,
            ),
        ],
    ),
    train_config=dict(
        device="cuda",
        lr=1e-4,
        epoches=100,
        last_epoch=60,
        restore=False,
        model_save_path="checkpoints/deeplabv3plus_tianchi_2.pkl",
        n_classes=5,
        mode="train",
    ),
    lr_scheduler=dict(step_size=10, gamma=0.5, last_epoch=-1),
)
