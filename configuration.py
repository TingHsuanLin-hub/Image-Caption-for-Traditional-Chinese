class Config(object):
    def __init__(self):

        # Learning Rates
        self.lr_backbone = 1e-5
        self.lr = 1e-4

        # Epochs
        self.epochs = 100
        self.lr_drop = 20
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Basic
        self.device = 'cuda'
        self.seed = 42
        self.batch_size = 100
        self.num_workers = 8
        self.checkpoint = './checkpoint.pth'
        self.clip_max_norm = 0.1

        # Transformer
        self.hidden_dim = 256 # must be consistent with embedding size(256 before)
        self.pad_token_id = 0
        self.max_position_embeddings = 32 # I think 32 will be safer.
        self.layer_norm_eps = 1e-12
        self.dropout = 0.1
        self.vocab_size = 256

        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.nheads = 8
        self.pre_norm = True

        # Dataset
        self.dir = '../coco'
        self.limit = -1
        self.wor2vec_model = './Word2Vec_model/word2vec_256caption_for_data_with_start.model'
        self.swin_transformer = './models/swinv2_tiny_patch4_window8_256.pth'
