BootStrap: docker
From: nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


%post
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this will install all necessary packages and prepare the container
    CUDNN_VERSION=7.0.5.15
    apt-get -y update --fix-missing

    # install cuDNN version 7.0.5 required for keras
    # apt-get install -y --no-install-recommends \
    #    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    #    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    #rm -rf /var/lib/apt/lists/*
    #apt-get -y update

    # install other dependencies
    apt-get -y install --allow-downgrades --no-install-recommends \
        build-essential \
        dbus \
        wget \
        git \
        mercurial \
        subversion \
        vim \
        nano \
        cmake \
        bzip2 \
        ca-certificates \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libboost-all-dev

#    locale-gen en_US
#    locale-gen en_US.UTF-8
#    locale update

#    system-machine-id-setup
    rm /etc/machine-id
    dbus-uuidgen --ensure=/etc/machine-id

    export CUDA_HOME="/usr/local/cuda"
    export CPATH="$CUDA_HOME/include:$CPATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    export PATH="$CUDA_HOME/bin:$PATH"
    export PATH="/opt/conda/bin:$PATH"

    # required for LightGBM
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

    export BOOST_ROOT=/usr/local/boost

    wget --quiet https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -O ~/anaconda.sh
    /bin/bash ~/anaconda.sh -b -p /opt/conda
    rm ~/anaconda.sh

    conda update conda
    conda install \
        spyder==3.2.6 \
        qtconsole==4.3.1 \
        qtpy==1.3.1
    pip install --upgrade pip
    pip install future

    # install tensorflow with gpu support
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.6.0-cp36-cp36m-linux_x86_64.whl

    # install tflearn
    pip install tflearn

    # install keras
    pip install keras

    # install pytorch (with Caffe2)
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # See https://pytorch.org/get-started/locally/
    conda install google-sparsehash -c bioconda
    conda install -c anaconda pillow
    git clone git@github.com:facebookresearch/SparseConvNet.git
    cd SparseConvNet/
    sed -i 's/torch.cuda.is_available()/True/g' setup.py
    bash develop.sh

    conda install numpy pandas torchnet matplotlib
    pip install scikit-learn Biopython seaborn tqdm dask joblib torchnet fastparquet pyarrow
    pip install dask[dataframe]

    # install OpenCV
    pip install opencv-python

    conda clean --index-cache --tarballs --packages --yes

%runscript
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this text code will run whenever the container
# is called as an executable or with `singularity run`
exec python $@

%help
This container is backed by Anaconda version 4.4.0 and provides the Python 3.6 bindings for:
    * Tensorflow 1.6.0
    * Keras 2.1.5
    * PyTorch 1.0
    * Caffe2
    * XGBoost
    * LightGBM
    * OpenCV
    * CUDA 9.0
    * CuDNN 7.0.5.15


%environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This sets global environment variables for anything run within the container
    export CUDA_HOME="/usr/local/cuda"
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
