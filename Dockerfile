FROM nvidia/cuda:10.2-cudnn7-devel

ARG function
ARG output
ARG input
ARG processCount

RUN apt-get update -y && \
	apt-get install -y --no-install-recommends \
	python3-dev \
	python3-pip \
	python3-wheel \
	python3-setuptools && \
	rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir -U install setuptools pip
RUN pip3 install --no-cache-dir cupy-cuda102==9.0.0b1 scipy optuna
RUN pip3 install numpy
RUN pip3 install opencv-python
COPY $input .
COPY UMatFileVideoStream.py .
COPY main.py .
CMD ["python3", "main.py", $input, "--function", $function, "--output", $output, "--process", $processCount]
