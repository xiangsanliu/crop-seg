# segformer_b4_tianchi_2_label_random_no_overlap
# 不重叠，全部打乱

dataset_path = "dataset/gaofen"

config = dict(
    model=dict(
        type="Segformer",
        model_config=dict(
            encode_config=dict(type="mit_b5", pretrained="pretrained/mit_b5.pth"),
            decoder_config=dict(
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                embed_dim=768,
                num_classes=2,
                dropout_ratio=0.1,
            ),
        ),
    ),
    train_pipeline=dict(
        dataloader=dict(
            batch_size=7, num_workers=7, drop_last=True, pin_memory=True, shuffle=True
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
            csv_file=f"{dataset_path}/val.csv",
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
        last_iou=0.9150184048800727,
        restore=False,
        model_save_path="checkpoints/Segformer_b4_tianchi_2.pkl",
        mode="train",
        n_classes=2
    ),
    lr_scheduler=dict(step_size=10, gamma=0.5, last_epoch=-1),
)
