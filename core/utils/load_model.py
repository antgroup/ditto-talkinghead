import inspect


def _create_module(module, device, kwargs):
    kwargs = dict(kwargs)
    params = inspect.signature(module).parameters
    accepts_device = "device" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_device:
        kwargs["device"] = device
    return module(**kwargs)


def load_model(model_path: str, device: str = "cuda", **kwargs):
    if kwargs.get("force_ori_type", False):
        # for hubert, landmark, retinaface, mediapipe
        model = load_force_ori_type(model_path, device, **kwargs)
        return model, "ori"

    if model_path.endswith(".onnx"):
        # onnx
        import onnxruntime

        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        model = onnxruntime.InferenceSession(model_path, providers=providers)
        return model, "onnx"

    elif model_path.endswith(".engine") or model_path.endswith(".trt"):
        # tensorRT
        from .tensorrt_utils import TRTWrapper

        model = TRTWrapper(model_path)
        return model, "tensorrt"

    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        # pytorch
        model = create_model(model_path, device, **kwargs)
        return model, "pytorch"

    else:
        raise ValueError(f"Unsupported model file type: {model_path}")


def create_model(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..models.modules",
    **kwargs,
):
    import importlib

    # module = getattr(importlib.import_module('..models.modules', __package__), module_name)
    module = getattr(importlib.import_module(package_name, __package__), module_name)
    # from <package_name> import <module_name>

    model = _create_module(module, device, kwargs)
    model.load_model(model_path).to(device)
    return model


def load_force_ori_type(
    model_path: str,
    device: str = "cuda",
    module_name="",
    package_name="..aux_models.modules",
    force_ori_type=False, 
    **kwargs,
):
    import importlib

    module = getattr(importlib.import_module(package_name, __package__), module_name)
    model = _create_module(module, device, kwargs)
    return model
