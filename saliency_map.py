import cv2
import numpy as np
import torch
from face_umb import get_dataloader


trainloader = get_dataloader("/mnt/umd_face", 0, 3, False, 2048, 2)
trainloader_iter = enumerate(trainloader)
_, batch = trainloader_iter.__next__()

images_cpu = batch[0]

def saliency_map(images):
    labels = torch.empty((images_cpu.shape[0],images_cpu.shape[2],images_cpu.shape[3]))
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    saliency_2 = cv2.saliency.StaticSaliencyFineGrained_create()

    for i, img in enumerate(images):
        img = img.permute(1, 2, 0)
        img = img.to('cpu').detach().numpy()
        bool, map = saliency.computeSaliency(img)
        i_saliency = (map * 255).astype("uint8") # 0-255 こっちを使用する
        cv2.imwrite('1.png', i_saliency)
        i_saliency = ((i_saliency / 255) - 0.5) * 2 # 正規化
        labels[i] = torch.from_numpy(i_saliency.astype(np.float32)).clone()

        # i_threshold = cv2.threshold(i_saliency, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # [0, 1]の2値
    return labels

labels = saliency_map(images_cpu)
