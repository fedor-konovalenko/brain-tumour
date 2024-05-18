# Model for brain tumour classification and tumour detection (MRI images)

## Research Purposes

It is necessary to develop a model that determines the type and location of the tumor (if present) in an MRI image of the brain. 
[Source Dataset](https://www.kaggle.com/datasets/sayemprodhanananta/brain-mri-dataset-containing-8-classes)
____________

## Models

### Classification
-------
There were several models, such as ResNet50, DenseNet121, MobilenetV2 with similar results (accuracy about 0.99, ROC-AUC about 1.0).

For the application the ResNet50 was selected.

___________________

### Detection

Then about 1100 images were marked with bounding boxes (with CVAT) and the SSD300-VGG16 model was trained.

**Results**

- maximum mAP50 value - 1,00
- mean mAP50 value - 0,81
- maximum IoU value - 0,97
- mean IoU value - 0,62

And with best models the FastAPI application were developed.



