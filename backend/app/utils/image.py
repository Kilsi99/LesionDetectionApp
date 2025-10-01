import numpy as np
import cv2
import torchvision.transforms as T
import torch.nn.functional as F
import torch

def preprocess_image(img, size=(256,256)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def overlay_mask(image, mask, colour=(0, 255, 0), alpha=0.5):
    """
    Overlay lesion mask onto image without blue tint.
    
    image: PIL.Image in RGB
    mask: torch tensor or numpy array, shape (H, W) or (1, H, W)
    colour: tuple in RGB
    alpha: transparency
    """
    import numpy as np
    import torch

    # Convert PIL → NumPy (RGB)
    image = np.array(image).astype(np.uint8)

    # Ensure mask is 2D binary
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)
    mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    mask = (mask > 0).astype(np.uint8)

    # Make a colored mask
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(3):
        colored_mask[:, :, i] = mask * colour[i]

    # Blend using broadcasting (avoids channel misalignment)
    overlay = image.copy()
    overlay = np.where(mask[:, :, None] == 1,
                       (1 - alpha) * image + alpha * colored_mask,
                       image).astype(np.uint8)

    return overlay
