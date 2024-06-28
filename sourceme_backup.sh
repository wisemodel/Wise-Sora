export HOME=/home/lijunjie
source /home/lijunjie/miniconda3/bin/activate
conda activate pixartpp
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/usr/local/cuda/bin' | tr '\n' ':')
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
