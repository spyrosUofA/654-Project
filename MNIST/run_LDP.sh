conda create --name myenv
y
conda activate myenv

cd Documents/GitHub/Federated-Learning-PyTorch/src


python3 localDP_main.py --seed 1 --norm_bound 1 --noise_scale 0.8
python3 localDP_main.py --seed 2 --norm_bound 1 --noise_scale 0.8
python3 localDP_main.py --seed 3 --norm_bound 1 --noise_scale 0.8
python3 localDP_main.py --seed 4 --norm_bound 1 --noise_scale 0.8
python3 localDP_main.py --seed 5 --norm_bound 1 --noise_scale 0.8