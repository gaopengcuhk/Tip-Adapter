FROM continuumio/miniconda3

RUN apt update
RUN apt install -y git

RUN git clone https://github.com/gaopengcuhk/Tip-Adapter

WORKDIR /Tip-Adapter
# RUN conda update -n base -c defaults conda
RUN conda install -y pytorch torchvision cudatoolkit -c pytorch -c conda-forge
RUN pip install -r requirements.txt


# CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml
CMD [ "CUDA_VISIBLE_DEVICES=0", "python", "main_imagenet.py", "--config", "configs/imagenet.yaml", "zs_gpt_v" ]

# Run Notes:
# `--shm-size 8G` may be required if you run out of shared memory
# `--gpus all` may be required if you have multiple GPUs
