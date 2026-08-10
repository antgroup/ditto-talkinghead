import numpy as np
import torch
from ..utils.load_model import load_model
from ..utils.torch_utils import cuda_autocast


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, feature_3d, kp_source, kp_driving):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature_3d": feature_3d, "kp_source": kp_source, "kp_driving": kp_driving})
            self.model.infer()
            pred = self.model.buffer["out"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), cuda_autocast(self.device):
                pred = self.model(
                    torch.from_numpy(feature_3d).to(self.device), 
                    torch.from_numpy(kp_source).to(self.device), 
                    torch.from_numpy(kp_driving).to(self.device)
                ).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return pred
