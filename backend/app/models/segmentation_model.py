import segmentation_models_pytorch as smp 

def segmentation_model():
    return smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )