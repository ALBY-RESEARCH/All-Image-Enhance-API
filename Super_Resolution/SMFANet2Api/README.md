# 📖 Api for SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution

Refer to this [README.md](https://github.com/Zheng-MJ/SMFANet/README.md) for more Detailed


---

### Requirements
> - Python 3.8, PyTorch >= 1.8
> - BasicSR 1.4.2
> - Platforms: Ubuntu 18.04, cuda-11(optional)

### Installation
```
# clone this repo
git clone https://github.com/ALBY-RESEARCH/All-Image-Enhance-API.git
cd All-Image-Enhance-API
git checkout phananh285
cd Super_Resolution/SMFANet2Api
# Install dependent packages
conda create --name smfan python=3.8
conda activate smfan
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```
You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation
### Data Preparation
Please refer to [datasets/REDAME.md](datasets/README.md) for data preparation.


### Testing 
- Download the testing dataset. 
- Run the following commands:
```
# test SMFANet for x4 efficient SR
python Main.py --input_dir YourDatasetPath --model smfan
Change YourDatasetPath accordingly to your end
```
- The test results will be in './results'.


### Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

### Contact
If you have any questions, please feel free to reach me out at mingjunzheng@njust.edu.cn

