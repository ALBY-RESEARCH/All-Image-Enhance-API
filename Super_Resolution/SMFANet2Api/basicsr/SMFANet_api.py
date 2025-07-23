import logging
import torch
from os import path as osp
import time
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.OptionAPI import dict2str, parse_options
import yaml
import os
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--in_yaml', type=str, required=True, help='Đường dẫn file YAML gốc')
#     parser.add_argument('--out_yaml', type=str, default='updated_config.yml', help='YAML sau khi cập nhật')
#     parser.add_argument('--input_dir', type=str, required=True, help='Đường dẫn ảnh input (LQ)')
#     parser.add_argument('--gt_dir', type=str, required=True, help='Đường dẫn ảnh ground truth (GT)')
#     args = parser.parse_args()

#     update_yaml(args.in_yaml, args.out_yaml, args.input_dir, args.gt_dir)


class SMFANetApi:
    '''
    SMFANet API for image super-resolution.
    This class provides methods to update the YAML configuration file and run the test pipeline.
    initialized with the input directory and options.
    Args:
        input_dir (str): Directory containing input images.
        opt (str): Path to the YAML configuration file.
        is_train (bool): Whether the API is used for training or testing. Default is False  
        root_path (str): Root path of the project. Default is the parent directory of this file.
    '''
    def __init__(self, input_dir,opt, is_train=False, root_path=osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))):
        self.root_path = root_path
        self.opt = opt
        self.input_dir = input_dir
        self.is_train = is_train
        
    def update_yaml(self, output_yaml):
        # Đọc YAML gốc
        with open(self.opt, 'r') as f:
            cfg = yaml.safe_load(f)

        # Cho người dùng nhập thêm nếu muốn
        # self.input_dir= input("Nhập đường dẫn ảnh input (LQ): ").strip()
        # new_gt = input("Nhập đường dẫn ground truth (GT): ").strip()

        # Ghi đè
        cfg['datasets']['test_1']['dataroot_lq'] = self.input_dir
        # cfg['datasets']['test']['dataroot_gt'] = new_gt

        # Ghi file YAML mới
        with open(output_yaml, 'w') as f:
            yaml.dump(cfg, f)
                       # Cập nhật đường dẫn để dùng luôn file mới
        print(f"✅ Đã tạo file YAML mới tại: {output_yaml}")
        self.opt = output_yaml
        return self.opt
    def test_pipeline(self):
       
        print("Có muốn cập nhật file YAML không? (y/n)")
                # Tạo tên file YAML mới

        update_choice = input().strip().lower()
        if update_choice == 'y':
            output_yaml = input("Nhập đường dẫn thư mục để lưu file YAML mới: ").strip()
            
            # Tạo thư mục nếu chưa tồn tại
            if not os.path.exists(output_yaml):
                os.makedirs(output_yaml)
                print(f"✅ Đã tạo thư mục: {output_yaml}")
            else:
                print(f"📂 Thư mục đã tồn tại: {output_yaml}")
            output_yaml = os.path.join(output_yaml, 'yournewconfig.yml')
            self.opt=self.update_yaml(output_yaml=output_yaml)
        else:
            print("🚫 Bỏ qua cập nhật file YAML, sử dụng file gốc.")

        # parse options, set distributed setting, set ramdom seed
        opt = parse_options(self.root_path, opt=self.opt, launcher='none', auto_resume=False, debug=False, local_rank=0, force_yml=None, is_train=self.is_train)
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

        # mkdir and initialize loggers
        make_exp_dirs(opt)
        log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(opt))

        # create test dataset and dataloader
        test_loaders = []
        for _, dataset_opt in sorted(opt['datasets'].items()):
            test_set = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)

        # create model
        model = build_model(opt)
        num_tested = 0
        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            num_tested += 1

            start = time.time()

            model.validation(
            test_loader, 
            current_iter=opt['name'], 
            tb_logger=None, 
            save_img=opt['val']['save_img']
            )

            end = time.time()
            elapsed = end - start
            logger.info(f'➤ Total inference time for {test_loader.dataset.opt["name"]}: {elapsed:.4f} seconds')

            # Nếu bạn chỉ muốn tính thời gian trung bình 1 ảnh:
            num_images = len(test_loader.dataset)
            avg_time = elapsed / num_images
            logger.info(f'⏱ Avg time per image: {avg_time:.4f} seconds')
            if num_tested >= 1:
                break  # NEW → chỉ chạy đúng 1 ảnh



# if __name__ == '__main__':
    # root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
#     api = SMFANet2Api(root_path)
#     api.test_pipeline()
