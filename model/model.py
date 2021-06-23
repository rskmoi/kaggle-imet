from pretrainedmodels.models import senet
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from timm import create_model


class SwinTransformerWithClassifier(torch.nn.Module):
    def __init__(self):
        super(SwinTransformerWithClassifier, self).__init__()
        self._swin  = create_model(model_name=f'swin_small_patch4_window7_224', pretrained=True)
        self._swin.head = torch.nn.Identity()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(768, 1103)

    def forward(self, x):
        x = self._swin.forward_features(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class SEResnext50WithClassifier(torch.nn.Module):
    def __init__(self):
        super(SEResnext50WithClassifier, self).__init__()
        self.se_resnext50 = senet.se_resnext50_32x4d(pretrained="imagenet")
        self.se_resnext50.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.se_resnext50.last_linear = torch.nn.Linear(512 * 4, 2048)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2048, 1103)

    def forward(self, x):
        x = self.se_resnext50(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class SEResnext101WithClassifier(torch.nn.Module):
    def __init__(self):
        super(SEResnext101WithClassifier, self).__init__()
        self.se_resnext101 = senet.se_resnext101_32x4d(pretrained="imagenet")
        self.se_resnext101.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.se_resnext101.last_linear = torch.nn.Linear(512 * 4, 2048)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2048, 1103)

    def forward(self, x):
        x = self.se_resnext101(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class SEnet154WithClassifier(torch.nn.Module):
    def __init__(self):
        super(SEnet154WithClassifier, self).__init__()
        self.senet154 = senet.senet154(pretrained="imagenet")
        self.senet154.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.senet154.last_linear = torch.nn.Linear(512 * 4, 2048)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(2048, 1103)

    def forward(self, x):
        x = self.senet154(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

def get_model(model_name, pretrained_model_path=None, multi=False):
    if model_name == "se_resnext50":
        model = SEResnext50WithClassifier()
    elif model_name == "se_resnext101":
        model = SEResnext101WithClassifier()
    elif model_name == "senet154":
        model = SEnet154WithClassifier()
    elif model_name == "swin":
        model = SwinTransformerWithClassifier()
    else:
        raise ValueError()

    model.to(DEVICE)

    if multi:
        model = torch.nn.DataParallel(model)

    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))

    return model