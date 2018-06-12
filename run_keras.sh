#! /bin/bash

set -e

singularity_img="/project/ppi-workspace/molmimic/edraizen-SingularityTorch-master-latest.simg"
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
        flask)
            cmd="flask"
            shift
            break
            ;;
        *)
            exit 1
            ;;
    esac
done

export SINGULARITY_BINDPATH="/project"
export PYTHONPATH="/project/ppi-workspace/molmimic:$PYTHONPATH"

module load singularity > /dev/null 2>&1

case "$cmd" in
    ipython)
        singularity exec --nv "${singularity_img}" ipython "$@"
        ;;
    python)
        singularity exec --nv "${singularity_img}" /anaconda/bin/python "$@"
        ;;
    notebook)
        singularity exec --nv "${singularity_img}" jupyter notebook "$@"
        ;;
    shell)
        singularity shell "${singularity_img}" "$@"
        ;;
    flask)
        export FLASK_APP=/project/ppi-workspace/molmimic/protein_viewer/protein_viewer.py
        singularity exec --nv "${singularity_img}" /anaconda/bin/flask run
        ;;
    help)
        usage
        exit 0
        ;;
esac
