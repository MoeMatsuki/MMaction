conda activate open-mmlab

# ModuleNotFoundError: No module named 'mmengine'
pip install -U openmim
mim install mmengine
# AssertionError: MMCV==1.7.0 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0.
mim install "mmcv>=2.0.0"

mim install mmcv==1.7.0
mim install mmdet==2.27.0