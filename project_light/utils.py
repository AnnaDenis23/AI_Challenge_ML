import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMG_SIZE = 224

# Трансформации для обычного инференса
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Трансформации для TTA (Test Time Augmentation)
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_single(model, image: Image.Image, device,
                   use_tta=False, n_augmentations=5):
    """
    Предсказание для одного изображения.
    Возвращает: (predicted_class: int, probabilities: np.ndarray)
    """
    model.eval()

    if use_tta:
        predictions = []
        for _ in range(n_augmentations):
            img_tensor = tta_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()
                predictions.append(probs)
        avg_pred = np.mean(predictions, axis=0)
        return int(np.argmax(avg_pred)), avg_pred[0]

    img_tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    return int(np.argmax(probs)), probs