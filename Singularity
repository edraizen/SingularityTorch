BootStrap: docker
From: nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04

################################################################################
%labels
################################################################################
MAINTAINER Wolfgang Resch, Eli Driazen

################################################################################
%environment
################################################################################
export PATH=/usr/local/sbin:/usr/sbin:/sbin:/bin:/usr/bin:/usr/local/bin:/usr/local/cuda/bin:
export PYTHONPATH=/usr/share/pdb2pqr:

################################################################################
%post
################################################################################

###
### install keras + tensorflow + other useful packages
###
apt-get update
apt-get install -y wget libhdf5-dev graphviz locales python python-pip git xvfb python-vtk pdb2pqr python-pandas
locale-gen en_US.UTF-8
apt-get clean

wget ftp://ftp.cmbi.ru.nl/pub/software/dssp/dssp-2.0.4-linux-i386 -O /usr/local/bin/dssp
chmod a+x /usr/local/bin/dssp

wget ftp://ftp.icgeb.trieste.it/pub/CX/CX.c.gz -O /usr/local/bin/CX.c.gz
gunzip /usr/local/bin/CX.c.gz
gcc -o /usr/local/bin/cx /usr/local/bin/CX.c -lm
rm /usr/local/bin/CX.c

pip install --upgrade pip
pip install --upgrade numpy
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
pip install torchvision
pip install setuptools wheel Pillow scikit-learn matplotlib ipython==5.5.0
pip install h5py
pip install mayavi
pip install --upgrade notebook
pip install cython
pip install Biopython
pip install cffi

wget http://freesasa.github.io/freesasa-2.0.2.tar.gz
tar -xzf freesasa-2.0.2.tar.gz
cd freesasa-2.0.2
./configure CFLAGS=-fPIC --enable-python-bindings --disable-json --disable-xml
make && make install

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
