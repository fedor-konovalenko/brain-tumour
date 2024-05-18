import PIL
import torch
import torchvision.transforms.v2 as transforms
import logging
from PIL import Image
import warnings
from torchvision.models import resnet50
from torch import nn

warnings.filterwarnings("ignore")

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

DEVICE = "cpu"

CLASSES = {0: 'no_tumor',
           1: 'glioma',
           2: 'meningioma',
           3: 'pituitary'
           }


def pretrained_resnet(params_path: str, device: str):
    """load model and weights"""
    model = resnet50(weights=None)
    in_features = 2048
    out_features = 4
    model.fc = nn.Linear(in_features, out_features)
    model = model.to(device)
    model.load_state_dict(torch.load(params_path, map_location=torch.device(device)))
    return model


def predict(path: str, inp_size: int):
    """detecting function"""
    try:
        image = Image.open(path).convert('RGB')
    except PIL.UnidentifiedImageError:
        m_logger.error(f'something wrong with image')
        status = 'Fail'
        return status, path
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((inp_size, inp_size)),
    ])
    image = transformer(image)
    model = pretrained_resnet('resnet50.pth', DEVICE)
    m_logger.info(f'model loaded')
    with torch.no_grad():
        x = image.to(DEVICE).unsqueeze(0)
        predictions = model.eval()(x)
    result = int(torch.argmax(predictions, 1).cpu())
    m_logger.info(f'classification completed')
    status = 'OK'
    return status, CLASSES[result]
