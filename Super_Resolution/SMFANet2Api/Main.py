import basicsr.SMFANet_api
import argparse
import os
import sys
'''
        self.root_path = root_path
        self.opt = opt
        self.input_dir = input_dir
        self.is_train = is_train
'''
if __name__ == "__main__":
    # Lấy đường dẫn tuyệt đối của file đang chạy (Main.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    # parser.add_argument('-rootpath', type=str, required=False, help='Root directory')
    parser.add_argument('--type',type=str,required=True,default="DF2K_x4SR", help='choose which model to use')
    parser.add_argument('--model',type=str,required=True,help="choose which model to be used")
    # parser.add_argument('--opt', type=str,required=True, default= help='YAML sau khi cập nhật')
    parser.add_argument('--input_dir', type=str, required=True, help='path to desired images (only into folder)')
    parser.add_argument('--is_train', type=bool, required=False,default=False, help=')')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    # parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    # parser.add_argument('--auto_resume', action='store_true')
    # parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument(
    #     '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()
    opt=""
    yaml_path=""
    
    # Tạo đường dẫn tương đối đến file YAML

    if args.type not in ["SMFANet_DF2K_x4SR","SMFANet_DF2K_x2SR","SMFANet_DF2K_x3SR"] and args.model =='smfan':
        print("❌ Type does not belong to model")
        sys.exit(1)
    elif args.type in ["SMFANet_DF2K_x4SR","SMFANet_DF2K_x2SR","SMFANet_DF2K_x3SR"] and args.model !='smfan':
            # Tạo đường dẫn tương đối đến file YAML
        print("❌ Type does not belong to model")
        sys.exit(1)    
    else: 
        yaml_path = os.path.join(current_dir, 'options', 'test', f'{args.type}.yml')  
    #To-do: adding more
    opt=yaml_path
    SMAFNetAPi=basicsr.SMFANet_api.SMFANetApi(args.input_dir,opt)
    SMAFNetAPi.test_pipeline()
    # Chạy pipeline test
    # api.test_pipeline()