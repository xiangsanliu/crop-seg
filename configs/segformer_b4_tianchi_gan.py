config = dict(
    model=dict(
        type="Segformer",
        model_config=dict(
            encode_config=dict(type="mit_b4", pretrained="pretrained/mit_b4.pth"),
            decoder_config=dict(
                in_channels=[64, 128, 320, 512],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                embed_dim=768,
                num_classes=5,
                dropout_ratio=0.1,
            ),
        ),
    ),
    train_pipeline=dict(
        dataloader=dict(
            batch_size=8, num_workers=8, drop_last=True, pin_memory=True, shuffle=True
        ),
        dataset=dict(
            type="ConcatDataset",
            csv_file1="dataset/tianchi_no/train.csv",
            csv_file2="dataset/tianchi_synthesized/train_label.csv",
            image_dir1="dataset/tianchi_no/image",
            image_dir2="dataset/tianchi_synthesized/image",
            label_dir1="dataset/tianchi_no/label",
            label_dir2="dataset/tianchi_synthesized/lable",
        ),
        transforms=[
            dict(type="RandomCrop", p=1, output_size=(512, 512)),
            dict(type="RandomHorizontalFlip", p=0.5),
            dict(type="RandomVerticalFlip", p=0.5),
            # dict(type="ColorJitter",brightness=0.08,contrast=0.08,saturation=0.08,hue=0.08),
            dict(type="ToTensor",),
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
        total_steps=50000,
        eval_steps=1,
        restore=False,
        model_type="Segformer_b4",
        restore_path="work/models/beit_gaofen/2021-10-09T17:10:39.pkl",
        n_classes=5,
        mode="train",
        eval=False,
    ),
    lr_scheduler=dict(step_size=10, gamma=0.5, last_epoch=-1),
)
