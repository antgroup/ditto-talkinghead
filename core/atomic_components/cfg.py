import os
import pickle
import numpy as np


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _replace_device(cfg, device):
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k == "device":
                cfg[k] = device
            else:
                _replace_device(v, device)
    elif isinstance(cfg, list):
        for item in cfg:
            _replace_device(item, device)
    

def parse_cfg(cfg_pkl, data_root, replace_cfg=None):

    def _check_path(p):
        if os.path.isfile(p):
            return p
        else:
            return os.path.join(data_root, p)

    cfg = load_pkl(cfg_pkl)

    # ---
    # replace cfg for debug
    if isinstance(replace_cfg, dict):
        if replace_cfg.get("device"):
            _replace_device(cfg, replace_cfg["device"])
        for k, v in replace_cfg.items():
            if not isinstance(v, dict):
                continue
            for kk, vv in v.items():
                cfg[k][kk] = vv
    # ---

    base_cfg = cfg["base_cfg"]
    audio2motion_cfg = cfg["audio2motion_cfg"]
    default_kwargs = cfg["default_kwargs"]

    for k in base_cfg:
        if k == "landmark478_cfg":
            for kk in ["task_path", "blaze_face_model_path", "face_mesh_model_path"]:
                if kk in base_cfg[k] and base_cfg[k][kk]:
                    base_cfg[k][kk] = _check_path(base_cfg[k][kk])
        else:
            base_cfg[k]["model_path"] = _check_path(base_cfg[k]["model_path"])

    audio2motion_cfg["model_path"] = _check_path(audio2motion_cfg["model_path"])

    avatar_registrar_cfg = {
        k: base_cfg[k]
        for k in [
            "insightface_det_cfg",
            "landmark106_cfg",
            "landmark203_cfg",
            "landmark478_cfg",
            "appearance_extractor_cfg",
            "motion_extractor_cfg",
        ]
    }

    stitch_network_cfg = base_cfg["stitch_network_cfg"]
    warp_network_cfg = base_cfg["warp_network_cfg"]
    decoder_cfg = base_cfg["decoder_cfg"]
    
    condition_handler_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "use_emo",
            "use_sc",
            "use_eye_open",
            "use_eye_ball",
            "seq_frames",
        ]
    }

    lmdm_cfg = {
        k: audio2motion_cfg[k]
        for k in [
            "model_path",
            "device",
            "motion_feat_dim",
            "audio_feat_dim",
            "seq_frames",
        ]
    }

    w2f_type = audio2motion_cfg["w2f_type"]
    wav2feat_cfg = {
        "w2f_cfg": base_cfg["hubert_cfg"] if w2f_type == "hubert" else base_cfg["wavlm_cfg"],
        "w2f_type": w2f_type,
    }
    
    return [
        avatar_registrar_cfg,
        condition_handler_cfg,
        lmdm_cfg,
        stitch_network_cfg,
        warp_network_cfg,
        decoder_cfg,
        wav2feat_cfg,
        default_kwargs,
    ]


def print_cfg(**kwargs):
    for k, v in kwargs.items():
        if k == "ch_info":
            print(k, type(v))
        elif k == "ctrl_info":
            print(k, type(v), len(v))
        else:
            if isinstance(v, np.ndarray):
                print(k, type(v), v.shape)
            else:
                print(k, type(v), v)
