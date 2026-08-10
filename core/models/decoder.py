import numpy as np
import torch
from ..utils.load_model import load_model
from ..utils.torch_utils import cuda_autocast


class Decoder:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "SPADEDecoder",
        }
        self.model, self.model_type = load_model(model_path, device=device, **kwargs)
        self.device = device
        
    def __call__(self, feature):

        if self.model_type == "onnx":
            pred = self.model.run(None, {"feature": feature})[0]
        elif self.model_type == "tensorrt":
            self.model.setup({"feature": feature})
            self.model.infer()
            pred = self.model.buffer["output"][0].copy()
        elif self.model_type == 'pytorch':
            with torch.no_grad(), cuda_autocast(self.device):
                pred = self.model(torch.from_numpy(feature).to(self.device)).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        pred = np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255    # [h, w, c]
        
        return pred
