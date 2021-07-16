class TransConfig(object):

    def __init__(
        self,
        patch_size,
        in_channel,
        out_channel,
        sample_rate=4,
        embed_dim=768,
        num_hidden_layers=8,
        num_heads=6,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        img_size=(512, 512),
        drop_ratio=0.1,
        attention_drop_ratio=0.1,
        drop_path_ratio=0.1,
        proj_drop_ratio= 0.1,
        mlp_ratio=4,
        decode_features=[512, 256, 128, 64]
    ):
        self.sample_rate = sample_rate
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embed_dim = embed_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_patches = int(
            img_size[0]/patch_size[0] * img_size[1]/patch_size[1])
        self.drop_ratio = drop_ratio
        self.attention_drop_ratio = attention_drop_ratio
        self.drop_path_ratio = drop_path_ratio
        self.proj_drop_ratio = proj_drop_ratio
        self.mlp_ratio = mlp_ratio
        self.decode_features = decode_features
