# nvmedcontrol

Environment setup
```
conda create --name=py310  python=3.10
conda activate py310

conda install pytorch=2.3.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


pip install -U pip 
pip install -U lightning rich torch_fidelity torch_ema datasets diffusers hydra-core tensorboard
pip install -U numpy==1.26.4
pip install -U transformers
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U plotly
pip install -U diffusers
pip install -U lightning
pip install -U tensorboard

pip install -U monai[nibabel]
pip install -U einops
pip install -U lmdb
pip install -U mlflow
pip install -U clearml
pip install -U scikit-image
pip install -U pytorch-ignite
pip install -U pandas
pip install -U pynrrd
pip install -U gdown

pip install -U git+https://github.com/Project-MONAI/GenerativeModels.git 
```

Check torch has been compiled with CUDA
```
python -c 'import torch; print(torch.cuda.is_available())'
python -c 'import torch; torch.randn(1, device="cuda:0")'
python -c "import torch; import monai; from monai.config import print_config; print_config()"
python -c "import torch; import pytorch3d; from monai.config import print_config; print_config()"
```

Edit perceptual loss monai
```
/home/quantm/anaconda3/envs/py310/lib/python3.10/site-packages/monai/losses/perceptual.py
Line 32
From medical_resnet50_23datasets = "medical_resnet50_23datasets"
To medical_resnet50_23datasets = "medicalnet_resnet50_23datasets"
```
-----
```
git clone https://github.com/tmquan/nvmedcontrol
cd nvmedcontrol
python script/setup.py # install pytorch3d
```

Change paths of folders in conf/hparams.yaml to the located data
```
train_image3d_folders
train_image2d_folders
val_image3d_folders
val_image2d_folders
test_image3d_folders
test_image2d_folders
```

```
python main_inversion.py --config-name hparams
```