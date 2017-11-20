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
export PYTHONPATH=/usr/share/pdb2pqr:/anaconda/lib/python2.7:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

################################################################################
%post
################################################################################

###
### install keras + tensorflow + other useful packages
###
apt-get update
apt-get install -y wget libhdf5-dev graphviz locales python python-pip git xvfb python-vtk pdb2pqr python-pandas curl
locale-gen en_US.UTF-8
apt-get clean

curl -LO "https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh"
bash ./Anaconda2-5.0.1-Linux-x86_64.sh -b -p /anaconda
/anaconda/bin/conda install pytorch torchvision cuda80 -c soumith


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
#/anaconda/bin/conda install cffi

wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
tar -xzf freesasa-2.0.2.tar.gz
cd freesasa-2.0.2
./configure CFLAGS=-fPIC --enable-python-bindings --with-python=/anaconda/bin/python --disable-json --disable-xml
make && make install

echo "Torch can see GPUs"
python -c "import torch; print torch.cuda.is_available()" 2>/dev/null

cd /
apt-get install libsparsehash-dev
git -c http.sslVerify=false clone http://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet/PyTorch/
python setup.py develop
pip install git+https://github.com/pytorch/tnt.git@master

###
### destination for NIH HPC bind mounts
###
mkdir /gpfs /spin1 /gs2 /gs3 /gs4 /gs5 /gs6 /gs7 /gs8 /data /scratch /fdb /lscratch /pdb
