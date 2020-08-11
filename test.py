import torch
import os
import cv2

from smoke.config import cfg
from smoke.data import make_data_loader
from smoke.solver.build import make_optimizer, make_lr_scheduler
from smoke.utils.check_point import DetectronCheckpointer
from smoke.engine import default_argument_parser, default_setup,launch
from smoke.utils import comm
from smoke.engine.trainer import do_train
from smoke.modeling.detector import build_detection_model
from smoke.engine.test_net import run_test


def setup(args):
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg

def load_checkpoint(model:torch.nn.Module, model_path:str, pdb=False):
    if not os.path.exists(model_path):
        return None
    pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)["model"]
    pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items()}

    if pdb:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys() and model_dict[k].size() == v.size()}

        import pdb; pdb.set_trace()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(pretrained_dict)
        
    model.eval()
    return model

def main(args):
    cfg = setup(args)

    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    model = load_checkpoint(model, "./tools/logs/model_final.pth")

    images = cv2.imread("./figures/test.jpg", 1)
    images = cv2.resize(images, (512,512))
    images = torch.from_numpy(images).float().cuda()
    images = images.permute(2,0,1).unsqueeze(0)
    with torch.no_grad():
        output = model(images, )
    import pdb; pdb.set_trace()

    return run_test(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)