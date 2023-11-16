FROM continuumio/miniconda3

RUN apt update
RUN apt install -y git

# RUN git clone https://github.com/gaopengcuhk/Tip-Adapter
COPY . /Tip-Adapter

WORKDIR /Tip-Adapter
# RUN conda update -n base -c defaults conda
RUN conda install -y pytorch torchvision cudatoolkit -c pytorch -c conda-forge
RUN pip install -r requirements.txt

# Caltech101 - https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp/view?usp=share_link
WORKDIR /Tip-Adapter/caltech-101
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp" -O 101_ObjectCategories.tar.gz && rm -rf /tmp/cookies.txt
RUN tar -xf 101_ObjectCategories.tar.gz && rm 101_ObjectCategories.tar.gz

# Caltech101 Labels json - https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN' -O split_zhou_Caltech101.json

WORKDIR /Tip-Adapter

# CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml
# CMD [ "CUDA_VISIBLE_DEVICES=0", "python", "main_imagenet.py", "--config", "configs/imagenet.yaml", "zs_gpt_v" ]

# CUDA_VISIBLE_DEVICES=0 python main.py --config configs/caltech101.yaml
CMD [ "CUDA_VISIBLE_DEVICES=0", "python", "main.py", "--config", "configs/caltech101.yaml", "zs_gpt_v" ]

# Run Notes:
# `--shm-size 8G` may be required if you run out of shared memory
# `--gpus all` may be required if you have multiple GPUs
