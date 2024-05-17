# nvmedcontrol

conda create --name=py310  python=3.10
conda activate py310

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers

pip install -U pip 
pip install -U pip  lightning rich torch_fidelity torch_ema datasets diffusers hydra-core tensorboard
pip install -U pip 
pip install -U transformers
pip install -U plotly
pip install -U diffusers
pip install -U lightning
pip install -U tensorboard

pip install -U monai[all]
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

python script/setup.py # install pytorch3d

python main_inversion.py --config-name hparams
