FROM pytorch/pytorch
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+cu117.html
RUN pip3 install -r requirements.txt
RUN pip3 install rdkit-pypi
RUN pip3 install deepchem
RUN pip3 install mlflow
RUN pip3 install streamlit
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip3 install git+https://github.com/ARM-software/mango.git
