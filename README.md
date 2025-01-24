# SafeDiffuser
Implementation of our ICLR 2025 paper - SafeDiffuser: Safe Planning for Diffusion Probabilistic Models （code to be uploaded soon）


![pipeline](imgs/safediffuser.png) 


# Installation
```
conda env create -f environment.yml
conda activate safediffuser
pip install -e .
pip install qpth cvxpy cvxopt
```

# Switch between different experiments
```
git switch maze2d/locomotion/kuka
```

# Important notes for SafeDiffusers

1. run the following code first before running any code to address potential bugs (mujoco200 is needed)
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-515
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/wei/.mujoco/mujoco200/bin
```
2. Choose diffusers/CG/Truncate/different safediffusers (maze2d case)
(1) Choose any one option betwee lines 1040 and 1095 of the file diffuser/models/diffusion.py
(2) Choose any one option betwee lines 300 and 358 of the file diffuser/utils/rendering.py   (for visualization purpose)
(3) Customize the diffusion output in the file diffuser/guides/policies.py  (__call__ function)
(4) Customize the diffusion saving/print in the files scripts/plan_maze2d.py (especially after line 55) and config/maze2d.py

# Use pre-trained models
## Downloading weights
Download pretrained diffusion models and value functions (from diffuser) with:
```
./scripts/download_pretrained.sh
```

## Planning
To plan with guided sampling, run:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained
```
The --logbase points the experiment loader to the folder containing the pretrained/self-trained models.

# Training from scratch
1. Train a diffusion model with:
```
python scripts/train.py --dataset halfcheetah-medium-expert-v2
```
The default hyperparameters are listed in locomotion:diffusion. You can override any of them with flags, eg, --n_diffusion_steps 100.

2. Train a value function with:
```
python scripts/train_values.py --dataset halfcheetah-medium-expert-v2
```
See locomotion:values for the corresponding default hyperparameters.

3. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:
```
python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```

# Reference
If you find this helpful, please cite our work:
```
@inproceedings{xiao2025safe,
  title = {SafeDiffuser: Safe Planning with Diffusion Probabilistic Models},
  author = {Wei Xiao and Tsun-Hsuan Wang and Chuang Gan and Ramin Hasani and Mathias Lechner and Daniela Rus},
  booktitle = {International Conference on Learning Representations},
  year = {2025}
}
```

# Acknowledgements
The diffusion model implementation and organization are based on Michael Janner's diffuser repo: https://github.com/jannerm/diffuser
