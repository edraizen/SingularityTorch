BootStrap: docker
From: nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 
#arcsUVA/anaconda:cuda10.0-cudnn7.4-py3.6

%post
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this will install all necessary packages and prepare the container
    apt-get -y update

    # install cuDNN and accessories
    apt-get install -y --no-install-recommends \
        build-essential \

    # install other tools and dependencies
    apt-get -y install --allow-downgrades --no-install-recommends \
        dbus \
        wget \
        git \
        vim \
        cmake \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libboost-all-dev

    rm /etc/machine-id
    dbus-uuidgen --ensure=/etc/machine-id

    export CUDA_HOME="/usr/local/cuda"
    export CPATH="$CUDA_HOME/include:$CPATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"

    export PATH="/opt/conda/bin:$PATH"
    #unset CONDA_DEFAULT_ENV
    export ANACONDA_HOME=/opt/conda
    
    #required for LightGBM
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

    export BOOST_ROOT=/usr/local/boost
    
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh
    /bin/bash ~/anaconda.sh -b -p /opt/conda
    rm ~/anaconda.sh
    
    conda update conda
    
    pip install --upgrade pip
    pip install future

    # conda update conda
    conda list
    pip install --upgrade \
        pip \
        future \
        protobuf \
        numpy \
        typing \
        hypothesis \
        pydot \
        opencv-python
        
        
    nvcc --version

    # install pytorch
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

    # install SparseConvNet
    git clone https://github.com/facebookresearch/SparseConvNet.git
    cd SparseConvNet/
    sed -i 's/assert/pass #/g' setup.py
    sed -i 's/torch.cuda.is_available()/True/g' setup.py
    rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
    python setup.py develop

    # install molmimic requirments
    pip install \
      pandas \
      tables \
      scikit-learn \
      Biopython \
      seaborn \
      tqdm \
      dask \
      dask[dataframe] \
      joblib \
      tornado==4.5.1 \
      toolz \
      partd >= 0.3.8 \
      cloudpickle >= 0.2.1 \
      tables \
      freesasa \
      boto3 \
      botocore \
      awscli \
      toil

    conda clean --index-cache --tarballs --packages --yes

%runscript
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this text code will run whenever the container
# is called as an executable or with `singularity run`
exec python $@

%help
This container is backed by Anaconda version 5.2.0 and provides the Python 3.6 bindings for:
    * PyTorch (latest)
    * Caffe2
    * OpenCV
    * CUDA 10.0
    * CuDNN 7.4

%environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This sets global environment variables for anything run within the container
export CUDA_HOME="/usr/local/cuda-10.0"
    export CPATH="$CUDA_HOME/include:$CPATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
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

