FROM tensorflow/tensorflow:2.3.1-gpu

RUN apt-get update && apt-get -y install \
        python3 \
        python3-dev \
        git \
        wget \
        unzip \
        cmake \
        build-essential \
        pkg-config \
        python3-pip \
        python3-setuptools

RUN pip3 install --upgrade pip

ADD requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# assuming is run in tesla with: docker run -dit --gpus all -v /home/pduque/phd/wmh_segmentation:/wmh_segmentation -v /usr/lib/x86_64-linux-gnu/nvidia/tesla-418/:/usr/lib/nvidia wmh_docker
ENV LD_LIBRARY_PATH=/usr/lib/nvidia/