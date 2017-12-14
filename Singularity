BootStrap: docker
From: nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

################################################################################
%labels
################################################################################
MAINTAINER Wolfgang Resch, Eli Driazen

################################################################################
%environment
################################################################################
export PATH=/anaconda/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin:/usr/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/usr/share/pdb2pqr:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

################################################################################
%post
################################################################################

echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

apt-get update
apt-get install -y wget libhdf5-dev graphviz locales python python-pip git pdb2pqr xvfb curl ca-certificates \
         libnccl2=2.0.5-2+cuda8.0 \
         libnccl-dev=2.0.5-2+cuda8.0 \
         libjpeg-dev \
         libpng-dev
locale-gen en_US.UTF-8
apt-get clean

curl -LO https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
chmod +x ./Anaconda2-5.0.1-Linux-x86_64.sh
bash ./Anaconda2-5.0.1-Linux-x86_64.sh -b -p /anaconda
rm ./Anaconda2-5.0.1-Linux-x86_64.sh
/anaconda/bin/conda remove --force numpy
/anaconda/bin/conda install numpy 
/anaconda/bin/conda install pytorch torchvision magma-cuda80 -c soumith

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

wget ftp://ftp.cmbi.ru.nl/pub/software/dssp/dssp-2.0.4-linux-i386 -O /usr/local/bin/dssp
chmod a+x /usr/local/bin/dssp

wget ftp://ftp.icgeb.trieste.it/pub/CX/CX.c.gz -O /usr/local/bin/CX.c.gz
gunzip /usr/local/bin/CX.c.gz
gcc -o /usr/local/bin/cx /usr/local/bin/CX.c -lm
rm /usr/local/bin/CX.c

/anaconda/bin/conda install scikit-learn 
#/anaconda/bin/conda install mayavi
/anaconda/bin/conda install cython
/anaconda/bin/conda install Biopython
/anaconda/bin/conda install -c anaconda flask
#/anaconda/bin/conda install -c electrostatics pdb2pqr

#FreeSASA can only install to defualt python, so get a copy of cython
pip install cython
#/anaconda/bin/activate
wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
tar -xzf freesasa-2.0.2.tar.gz
cd freesasa-2.0.2
./configure CFLAGS="-fPIC -O2" --enable-python-bindings --disable-json --disable-xml --prefix=`pwd`
make && make install

export PATH=/anaconda/bin:/usr/local/sbin:/usr/sbin:/sbin:/bin:/usr/bin:/usr/local/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

cd /
apt-get install libsparsehash-dev
git -c http.sslVerify=false clone http://github.com/edraizen/SparseConvNet.git
cd SparseConvNet/PyTorch/
/anaconda/bin/python setup.py develop

git -c http.sslVerify=false clone http://github.com/pytorch/tnt.git
cd tnt
/anaconda/bin/python setup.py develop


###
### destination for NIH HPC bind mounts
###
mkdir /gpfs /spin1 /gs2 /gs3 /gs4 /gs5 /gs6 /gs7 /gs8 /data /scratch /fdb /lscratch /pdb
