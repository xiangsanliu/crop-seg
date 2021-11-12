# segformer_b4_tianchi_2_label_random_no_overlap
# 不重叠，全部打乱

dataset_path = "dataset/gaofen"

config = dict(
    model=dict(
        type="hrnet",
        model_config=dict(
            num_classes=2
        ),
    ),
    train_pipeline=dict(
        dataloader=dict(
            batch_size=8, num_workers=8, drop_last=True, pin_memory=True, shuffle=True
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
        lr=1e-2,
        total_steps=80000,
        eval_steps=1000,
        last_step=10000,
        restore=False,
        restore_path="work/models/segformer_b5_gaofen/2021-10-12T07:48:04.pkl",
        n_classes=2,
    ),
    lr_scheduler=dict(step_size=5, gamma=0.5, last_epoch=-1),
)
