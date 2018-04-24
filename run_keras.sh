#! /bin/bash

# Author: Wolfgang Resch (staff@helix.nih.gov)

set -e

keras_img="PATH_TO_SIMG.simg"
keras_backend="tensorflow"
cmd="help"

while true; do
    case "$1" in
        ipython)
            cmd="ipython"
            shift
            break
            ;;
        python)
            cmd="python"
            shift
            break
            ;;
        notebook)
            cmd="notebook"
            shift
            break
            ;;
        shell)
            cmd="shell"
            shift
            break 
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

export SINGULARITY_BINDPATH="/gs3,/gs4,/gs5,/gs6,/gs7,/gs8,/gs11,/gpfs,/spin1,/data,/pdb,/scratch,/fdb,/lscratch"
export KERAS_BACKEND="$keras_backend"
export HOSTNAME="cn4216"

module load singularity > /dev/null 2>&1

case "$cmd" in
    ipython)
        singularity exec --nv "${keras_img}" ipython "$@"
        ;;
    python)
        singularity exec --nv "${keras_img}" /anaconda/bin/python "$@"
        ;;
    notebook)
        singularity exec --nv "${keras_img}" jupyter notebook "$@" 
        ;;
    shell)
        singularity shell "${keras_img}" "$@"
        ;;
    help)
        usage
        exit 0
        ;;
esac
