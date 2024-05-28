# Wise-Sora
## Installation
1. Clone this repository and navigate to Wise-Sora folder
```
git clone https://github.com/wisemodel/Wise-Sora.git
cd Wise-Sora
```
2. Install required packages
```
conda create -n wisesora python=3.8 -y
conda activate wisesora
pip install -r requirements.txt
```
## Demo
### CausalVideoVAE
1. download weights
Download the weights from [HF](https://huggingface.co/supermodelteam/autoencoder/tree/main/v1), and then assign the weighs directory to the `--model_path` parameter of the `rec_video.sh` script in the directory `wisesora/autoencoder/causalvae`. 
2. setup environment (optional)
```
export PYTHONPATH=$PYTHONPATH:/path/to/Wise-Sora/wisesora/autoencoder/causalvae
```
3. run demo  
Now you can run the video reconstruction demo as follows.
```
cd wisesora/autoencoder/causalvae
bash rec_video.sh
```
## Train
### CausalVideoVAE
on the way