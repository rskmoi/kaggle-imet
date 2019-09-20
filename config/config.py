import yaml
from easydict import EasyDict


def _get_default_config():
    c = EasyDict()

    # dataset
    c.data = EasyDict()
    c.data.tta_zoom = 1.0
    c.data.img_size = 256
    c.data.train_csv = "./data/train_split_90pc.csv"
    c.data.valid_csv = "./data/valid_split_10pc.csv"
    c.data.test_dir = "../input/test"
    c.data.params = EasyDict()

    # model
    c.model = EasyDict()
    c.model.name = "se_resnext50"
    c.model.pretrained_model_path = None
    c.model.multi = False
    c.model.params = EasyDict()

    # train
    c.train = EasyDict()
    c.train.batch_size = 128
    c.train.num_epochs = 50
    c.train.lr = 1e-4

    # evaluation
    c.eval = EasyDict()
    c.eval.batch_size = 64

    # path
    c.path = EasyDict()
    c.path.out = "./result/out"
    return c


def _merge_config(src, dst):
    if not isinstance(src, EasyDict):
        return

    for k, v in src.items():
        if isinstance(v, EasyDict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = EasyDict(yaml.load(fid))

    config = _get_default_config()
    _merge_config(yaml_config, config)
    return config
