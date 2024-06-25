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
You will get the following reconstructed video.
[![CausalVAE Demo](https://res.cloudinary.com/marcomontalbano/image/upload/v1719288459/video_to_markdown/images/youtube--FqCrvu6ZHzg-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=FqCrvu6ZHzg "CausalVAE Demo")  

More examples can be viewed from the [supermodelteam](https://www.youtube.com/@supermodelteam) channel on YouTube.
## Train
### CausalVideoVAE
on the way