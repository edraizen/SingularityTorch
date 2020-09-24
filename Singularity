BootStrap: docker
From: nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

%post
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this will install all necessary packages and prepare the container
    apt-get -y update
    apt-get -y upgrade

    # install other dependencies
    apt-get -y install --allow-downgrades --no-install-recommends \
        build-essential \
        dbus \
        wget \
        git \
        vim \
        nano \
        cmake \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libboost-all-dev \
        gdb \
        libopenblas-dev \
        g++-7

    rm /etc/machine-id
    dbus-uuidgen --ensure=/etc/machine-id

    export CUDA_HOME="/usr/local/cuda-10.2"
    export CPATH="$CUDA_HOME/include:$CPATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/opt/conda/lib:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
    export PATH="/opt/conda/bin:$PATH"

    #required for LightGBM
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

    export BOOST_ROOT=/usr/local/boost

    #wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    /bin/bash ~/miniconda.sh -b -p /opt/conda
    rm ~/miniconda.sh
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y python=3.7
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -c anaconda future
      
    # install SparseConvNet
    #git clone https://github.com/facebookresearch/SparseConvNet.git
    #cd SparseConvNet/
    #sed -i 's/assert/pass #/g' setup.py
    #sed -i 's/torch.cuda.is_available()/True/g' setup.py
    #rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
    #export TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5" 
    #python setup.py develop
   
    #. /opt/conda/etc/profile.d/conda.sh && \
    # conda install -y -c intel mkl mkl-include
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y -c pytorch magma-cuda102
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y -c anaconda cudatoolkit=10.2
    #. /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y numpy mkl-include pytorch cudatoolkit=10.2 -c pytorch
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y -c anaconda openblas
    
    cd /opt
    git clone --recursive https://github.com/pytorch/pytorch
    cd /opt/pytorch
    git checkout 1.6
    git submodule sync
    git submodule update --init --recursive
    
    export TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;3.5+PTX;3.7+PTX;5.0+PTX;5.2+PTX;5.3+PTX;6.0+PTX;6.1+PTX;6.2+PTX;7.0+PTX;7.2+PTX;7.5+PTX;8.0+PTX"
    
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    
    #https://github.com/pytorch/pytorch/issues/13541 # -D_GLIBCXX_USE_CXX11_ABI=0
    export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    
    #sed -i 's/set(TORCH_CXX_FLAGS/#set(TORCH_CXX_FLAGS/g' cmake/TorchConfig.cmake.in
    #sed -i 's/@GLIBCXX_USE_CXX11_ABI@/0/g' cmake/TorchConfig.cmake.in
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && pip install -v .
    
    #. /opt/conda/etc/profile.d/conda.sh && conda activate && pip install -U MinkowskiEngine
    git clone https://github.com/edraizen/MinkowskiEngine.git
    cd MinkowskiEngine
    ls -la $CUDA_HOME
    export CXX=gcc-7
    . /opt/conda/etc/profile.d/conda.sh && conda activate && python setup.py install --force_cuda #--blas=openblas

    # install requirements for molmimic
    pip install dask[dataframe]
    pip install scikit-learn Biopython seaborn tqdm dask joblib torchnet tables fastparquet pyarrow
    . /opt/conda/etc/profile.d/conda.sh && conda activate && pip install --ignore-installed boto3 botocore awscli toil
     . /opt/conda/etc/profile.d/conda.sh && conda activate && pip install --ignore-installed freesasa==2.0.3.post7
    pip install tensorboardX
    #pip install pytorch-lightning
    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
    pip install test-tube
    pip install wandb
    
    # install ipython and kernel to create a new jupyter kernal
    pip install ipython ipykernel

    # install OpenCV
    pip install opencv-python

    #conda clean --index-cache --tarballs --packages --yes

%runscript
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this text code will run whenever the container
# is called as an executable or with `singularity run`
exec python $@

%help
This container is backed by Anaconda version 4.4.0 and provides the Python 3.6 bindings for:
    * PyTorch 1.0
    * SparseConvNet
    * XGBoost
    * LightGBM
    * OpenCV
    * CUDA 9.0
    * CuDNN 7.0.5.15


%environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This sets global environment variables for anything run within the container
    export CUDA_HOME="/usr/local/cuda-10.2"
    export CPATH="$CUDA_HOME/include:$CPATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/opt/conda/lib:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"

    export PATH="/opt/conda/bin:$PATH"
    unset CONDA_DEFAULT_ENV
    export ANACONDA_HOME=/opt/conda

    XGBOOSTROOT=/opt/xgboost
    export CPATH="$XGBOOSTROOT/include:$CPATH"
    export LD_LIBRARY_PATH="$XGBOOSTROOT/lib:$LD_LIBRARY_PATH"
    export PATH="$XGBOOSTROOT:$PATH"
    export PYTHONPATH=$XGBOOSTROOT/python-package:$PYTHONPATH

    LIGHTGBMROOT=/opt/LightGBM
    export CPATH="$LIGHTGBMROOT/include:$CPATH"
    export LD_LIBRARY_PATH="$LIGHTGBMROOT:$LD_LIBRARY_PATH"
    export PATH="$LIGHTGBMROOT:$PATH"
    export PYTHONPATH=$LIGHTGBMROOT/python-package:$PYTHONPATH
