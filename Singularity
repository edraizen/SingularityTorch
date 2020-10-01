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
        libopenblas-dev
        
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
    
    apt-get -y install --allow-downgrades --no-install-recommends \
        libnccl2=2.7.8-1+cuda10.2\
        libnccl-dev=2.7.8-1+cuda10.2
        
    apt-get -y install software-properties-common
    add-apt-repository ppa:ubuntu-toolchain-r/test
    apt-get -y install g++-7
    
    # Install the IB verbs
    apt-get install -y --no-install-recommends libibverbs*
    apt-get install -y --no-install-recommends ibverbs-utils librdmacm* infiniband-diags libmlx4* libmlx5* libnuma*
    
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

    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    /bin/bash ~/miniconda.sh -b -p /opt/conda
    rm ~/miniconda.sh
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -y python=3.7
    
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install -c anaconda future
      
    . /opt/conda/etc/profile.d/conda.sh && conda activate && conda install numpy mkl-include pytorch cudatoolkit=10.2 -c pytorch
    
    #https://github.com/pytorch/pytorch/issues/13541 # -D_GLIBCXX_USE_CXX11_ABI=0
    #export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
    
    
    
    #. /opt/conda/etc/profile.d/conda.sh && conda activate && pip install -U MinkowskiEngine
    git clone https://github.com/edraizen/MinkowskiEngine.git
    cd MinkowskiEngine
    ls -la $CUDA_HOME
    export CXX=gcc-7
    . /opt/conda/etc/profile.d/conda.sh && conda activate && python setup.py install --force_cuda 
    
    # Install Open MPI
    #mkdir /tmp/openmpi && \
    #    cd /tmp/openmpi && \
    #    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    #    tar zxf openmpi-4.0.0.tar.gz && \
    #    cd openmpi-4.0.0 && \
    #    ./configure --enable-orterun-prefix-by-default && \
    #    make -j $(nproc) all && \
    #    make install && \
    #    ldconfig && \
    #    rm -rf /tmp/openmpii
    #export MPI_C_COMPILER=mpiicc
    #export MPI_CXX_COMPILER=mpiicpc
    #export MPI_CXX=mpiicpc
    #export -DMPI_CXX_LIBRARIES=[path to wherever the object is]/libmpi_cxx.so \
    #-DMPI_C_LIBRARIES=[path to wherever the object is]/libmpi.so \
    #-DMPI_CXX_INCLUDE_PATH=[path to wherever MPI headers are] \
    #-DMPI_C_INCLUDE_PATH=[same header path as above]
    #. /opt/conda/etc/profile.d/conda.sh && conda activate && conda install openmpi openmpi-mpicc
    #. /opt/conda/etc/profile.d/conda.sh && conda activate && conda uninstall intel-openmp
    apt-get install --reinstall openmpi-bin libopenmpi-dev
    which mpicc
    export MPI_CXX=mpicc

    # Install Horovod, temporarily using CUDA stubs
    ldconfig $CUDA_HOME/targets/x86_64-linux/lib/stubs && \
    HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig
    
    # Configure OpenMPI to run good defaults:
    #   --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
    echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf 
    #echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
    
    # Install OpenSSH for MPI to communicate between containers
    apt-get install -y --no-install-recommends openssh-client openssh-server && \
        mkdir -p /var/run/sshd

    # Allow OpenSSH to talk to containers without asking for confirmation
    cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
        echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
        mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

    # Set default NCCL parameters
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

    # install requirements for molmimic
    pip install dask[dataframe]
    pip install scikit-learn Biopython seaborn tqdm dask joblib torchnet tables fastparquet pyarrow
    . /opt/conda/etc/profile.d/conda.sh && conda activate && pip install --ignore-installed boto3 botocore awscli toil
     . /opt/conda/etc/profile.d/conda.sh && conda activate && pip install --ignore-installed freesasa==2.0.3.post7
    pip install tensorboardX
    #pip install pytorch-lightning
    pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade
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
