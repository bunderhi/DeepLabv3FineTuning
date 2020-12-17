""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
import models.segmentation.segmentation 
#from torchvision import models


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=True)
    # Added a Tanh activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model

def createFCN(outputchannels=1):
    model = models.segmentation.segmentation.fcn_resnet18(
        pretrained=False, progress=True)
    # Set the model in training mode
    model.train()
    return model