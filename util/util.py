import shutil
import os
import yaml
from easydict import EasyDict
import subprocess

# 以下はEasyDictをyaml出力するためのコード
def represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())
yaml.add_representer(EasyDict, represent_odict)


def make_output_dir(config, f):
    out = config.path.out
    if f:
        shutil.rmtree(out)
    assert not os.path.exists(out), "Output directory is already exists."
    os.makedirs(out)

    cur_hash = subprocess.run("git rev-parse HEAD", shell=True, stdout = subprocess.PIPE)
    config.git = EasyDict()
    config.git.hash = cur_hash.stdout.strip().decode("utf-8")
    with open(os.path.join(out, "all_config.yml"), "wb") as f:
        f.write(yaml.dump(config, default_flow_style=False).encode("utf-8"))