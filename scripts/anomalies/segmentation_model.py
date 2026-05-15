import torch
import numpy as np

class SegmentationModel:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.to(self.device)

    def load_model(self, model_path):
        model = torch.jit.load(model_path)
        model.eval()
        return model
    
    def predict(self, tile_image):
        input_array = tile_image.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(input_tensor)
            prob = torch.sigmoid(logit).squeeze().cpu()
            pred = (prob > 0.1)
            return pred.numpy()
