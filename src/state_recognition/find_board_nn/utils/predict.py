import torch
import torch.nn.functional as F

from .preprocess import preprocess


def predict_image(image, net, device, scale=0.5, out_threshold=0.5):
    net.eval()
    full_height, full_width = image.shape[0], image.shape[1]
    image = torch.from_numpy(preprocess(image, scale)).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(image).cpu()
        output = F.interpolate(output, (full_height, full_width), mode='bilinear')
        mask = torch.sigmoid(output) > out_threshold
    return mask.long().squeeze().numpy()
