dataset_path = "dataset/gaofen"

config = dict(
    model=dict(
        type="BEiT",
        model_config=dict(
            encoder_config=dict(
                pretrained="pretrained/beit_base_patch16_224_pt22k.pth",
                # pretrained="pretrained/beit_base_patch16_224_pt22k_ft22k.pth",
                img_size=512,
                patch_size=16,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                use_abs_pos_emb=False,
                use_rel_pos_bias=True,
                init_values=0.1,
                drop_path_rate=0.1,
                out_indices=[3, 5, 7, 11],
            ),
            decoder_config=dict(
                in_channels=[768, 768, 768, 768],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=768,
                dropout_ratio=0.1,
                num_classes=2,
                # norm_cfg=norm_cfg,
                align_corners=False,
            ),
        ),
    ),
    train_pipeline=dict(
        dataloader=dict(
            batch_size=6, num_workers=6, drop_last=True, pin_memory=True, shuffle=True
        ),
        dataset=dict(
            type="PNG_Dataset",
            csv_file=f"{dataset_path}/seg_train.csv",
            image_dir=f"{dataset_path}/image",
            mask_dir=f"{dataset_path}/label",
        ),
        transforms=[
            dict(type="RandomHorizontalFlip", p=0.5),
            dict(type="RandomVerticalFlip", p=0.5),
            # dict(type="RandomRot", p=0.5),
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
            csv_file=f"{dataset_path}/seg_val.csv",
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
        lr=1e-5,
        total_steps=80000,
        eval_steps=500,
        last_step=10000,
        restore=False,
        restore_path="work/models/beit_gaofen/2021-10-09T17:10:39.pkl",
        n_classes=2,
    ),
    lr_scheduler=dict(step_size=10, gamma=0.5, last_epoch=-1),
)
