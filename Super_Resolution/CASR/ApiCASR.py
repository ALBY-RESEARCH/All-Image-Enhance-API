import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pathlib
import logging
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import source.models
import source.utils.dataset as dd

from source.utils import util_logger
from source.utils import util_image as util
from source.utils.model_summary import get_model_flops
from source.models.get_model import get_model
import yaml
"""This script is used to deploy a super-resolution model for the All-Image-Enhance-API
        config : đường dẫn file config của mô hình
        checkpoint: đường dẫn đến mô hình sử dụng
        save dir: đường dẫn lưu kết quả
        submission_id: id của submission
        lr_dir: đường dẫn đến thư mục chứa ảnh LR
        save_sr: có lưu ảnh SR hay không
        scale: tỉ lệ phóng đại
        batch_size: kích thước batch
        num_workers: số lượng worker cho DataLoader
        cropsize: kích thước crop của ảnh
        bicubic: có sử dụng bicubic interpolation hay không
        fp16: có sử dụng fp16 hay không
"""
config="projects/All-Image-Enhance-API/Super_Resolution/CASR/pretrained_models/casr_weight/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/config.yml"
submission_id = "0002"
checkpoint="projects/All-Image-Enhance-API/Super_Resolution/CASR/pretrained_models/casr_weight/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/models/model_x4_best_submission_deploy.pt"
save_dir = "projects/All-Image-Enhance-API/Super_Resolution/CASR/WEIGHT_RESULT/Candidate/m4c64/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/"
lr_dir = "projects/All-Image-Enhance-API/Super_Resolution/CASR/testdata"
save_sr = True
scale = 4
batch_size = 1
num_workers = 1
crop_size = [540, 960]  # [720, 1280] or [1080, 2040]
bicubic = True
fp16 = True
def main(submission_id=submission_id, checkpoint=checkpoint, save_dir=save_dir, config=config, lr_dir=lr_dir, save_sr=save_sr, scale=scale, batch_size=batch_size, num_workers=num_workers, crop_size=crop_size, bicubic=bicubic, fp16=fp16):
    """
    SETUP DIRS
    """
    pathlib.Path(os.path.join(save_dir, "results")).mkdir(parents=True, exist_ok=True)

    """
    SETUP LOGGER
    """
    util_logger.logger_info("AIS2024-RTSR", log_path=os.path.join(save_dir, submission_id, f"Submission_{submission_id}.txt"))
    logger = logging.getLogger("AIS2024-RTSR")

    """
    BASIC SETTINGS
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config if provided
    if config is not None and os.path.isfile(config):
        with open(config) as f:
            yaml_args = yaml.load(f, Loader=yaml.FullLoader)
        # Update variables if present in config
        model_name = yaml_args.get('model', None)
        m_plainsr = yaml_args.get('m_plainsr', None)
        c_plainsr = yaml_args.get('c_plainsr', None)
        act_type = yaml_args.get('act_type', None)
        colors = yaml_args.get('colors', None)
        bias = yaml_args.get('bias', None)
        # ...add more if needed
    else:
        model_name = None
        m_plainsr = None
        c_plainsr = None
        act_type = None
        colors = None
        bias = None

    """
    LOAD MODEL
    """
    if not bicubic:
        # Tạo một namespace giả để truyền vào get_model
        class Cfg:
            pass
        cfg = Cfg()
        cfg.model = model_name
        cfg.m_plainsr = m_plainsr
        cfg.c_plainsr = c_plainsr
        cfg.act_type = act_type
        cfg.scale = scale
        cfg.colors = colors
        cfg.bias = bias
        model = get_model(cfg, device, mode='Deploy')
        if checkpoint is not None:
            model_path = os.path.join(checkpoint)
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('Params number: {}'.format(number_parameters))
        print(model)

    """
    SETUP DATALOADER
    """
    dataset = dd.SRDataset(lr_images_dir=lr_dir, n_channels=3, transform=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    """
    TESTING
    """
    with torch.no_grad():
        for img_L, img_path in tqdm(dataloader):
            img_name, ext = os.path.splitext(img_path[0])
            img_L = img_L.to(device, non_blocking=True)
            if bicubic:
                img_E = F.interpolate(img_L, scale_factor=scale, mode="bicubic", align_corners=False)
            else:
                if fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        img_E = model(img_L)
                else:
                    img_E = model(img_L)
            if save_sr:
                img_E = util.tensor2uint(img_E)
                util.imsave(img_E, os.path.join(save_dir, "results", img_name + ".png"))
            return img_E
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # specify submission
#     parser.add_argument("--submission-id", type=str, default='0002')
#     # parser.add_argument("--model-name", type=str, choices=["swin2sr", "imdn", "rfdn"], default='RepNetwork_V012_BestStruct')
#     parser.add_argument("--checkpoint", type=str, default='./pretrained_models/casr_weight/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/models/model_x4_best_submission_deploy.pt')
#     parser.add_argument("--save-dir", type=str, default="./WEIGHT_RESULT/Candidate/m4c64/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/")
#     parser.add_argument('--config', type=str, default='./pretrained_models/casr_weight/RepNetwork_V012_BestStruct_x4_p384_m1_c32_gelu_l2_adam_lr5e-05_e800_t2024-0326-0504/config.yml', help = 'pre-config file for training')
    
#     # specify dirs
#     parser.add_argument("--lr-dir", type=str, default='./datasets/RTSR/val_lr')
#     parser.add_argument("--save-sr", action="store_true", default=True)
    
#     # specify test case
#     parser.add_argument("--scale", type=int, default=4)
#     parser.add_argument("--batch-size", type=int, default=1)
#     parser.add_argument("--num-workers", type=int, default=1)
#     #parser.add_argument("--crop-size", type=int, nargs="+", default=[1080, 2040])
#     # parser.add_argument("--crop-size", type=int, nargs="+", default=[720, 1280])
#     parser.add_argument("--crop-size", type=int, nargs="+", default=[540, 960])
#     parser.add_argument("--bicubic", action="store_true",default=True)
#     parser.add_argument("--fp16", action="store_true", default=True)
#     args = parser.parse_args()
#     # tham so can : submission_id,check point, save dir, config, lr-dir, save-sr, scale, batch-size, num-workers, crop-size, bicubic, fp16 

#     main(args)