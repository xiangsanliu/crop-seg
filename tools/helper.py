import torch


def load_pretained(model, config):
    pre_trained_path = '/home/xiangjianjian/.cache/torch/hub/checkpoints/jx_vit_large_p16_384-b3be5167.pth'
    state_dict = torch.load(pre_trained_path, map_location=None)
    classifier_name = 'head'
    if config.num_classes != 1000:
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False
    del state_dict['pos_embed']
    model.load_state_dict(state_dict, strict=strict)

