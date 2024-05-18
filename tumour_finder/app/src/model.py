import PIL
import torch
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision import models
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_bounding_boxes, save_image
import logging
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

m_logger = logging.getLogger(__name__)
m_logger.setLevel(logging.DEBUG)
handler_m = logging.StreamHandler()
formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
handler_m.setFormatter(formatter_m)
m_logger.addHandler(handler_m)

IMG_SIZE = 320
DEVICE = "cpu"

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def tuned_ssd(params_path: str, num_classes: int, size: int, device: str):
    """load model and weights TODO : remove additional weights"""
    model = models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT,
                                          weights_backbone=models.VGG16_Weights,
                                          pretrained=False, score_thresh=.25)
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    # Image size for transforms.
    model.transform.min_size = (size,)
    model.transform.max_size = size
    model = model.to(device)
    model.load_state_dict(torch.load(params_path, map_location=torch.device(device)))
    return model


def evaluate(path: str):
    """detecting function"""
    try:
        image = Image.open(path)
    except PIL.UnidentifiedImageError:
        m_logger.error(f'something wrong with image')
        status = 'Fail'
        return status, path
    shape = (image.size[1], image.size[0])
    output_tr = transforms.Resize(shape)
    image = transformer(image)
    model = tuned_ssd('ssd.pth', 2, IMG_SIZE, DEVICE)
    m_logger.info(f'model loaded')
    with torch.no_grad():
        x = image.to(DEVICE)
        predictions = model.eval()([x, ])
    predictions = predictions[0]
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    label = [f"predicted fracture"] * len(predictions["labels"])
    boxes = predictions["boxes"].long()
    try:
        output_image = draw_bounding_boxes(image, boxes, fill=False, colors=['red']*len(label), width=2)
        output_image = output_tr(output_image)
        img_path = path.split('/')[0] + '/res_detect_' + path.split('/')[1]
        m_logger.info(f'detection completed')
        status = 'OK'
    except ValueError:
        status = 'Fail'
        output_image = torch.zeros([IMG_SIZE, IMG_SIZE])
        img_path = ''
        m_logger.error(f'detection failed')
    if status == 'OK':
        output_image = output_image.unsqueeze(0).permute(0, 1, 2, 3) / 255
        save_image(output_image, img_path)
        m_logger.info(f'image saved')
    return status, img_path
