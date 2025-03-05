# Pull base image.
FROM ubuntu:20.04

# Install.
ARG DEBIAN_FRONTEND=noninteractive

RUN \
  # Basic ubuntu stuff
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade
RUN \
  # Python
  apt install -y python3 && \
  apt-get install -y pip
RUN \
  # Python dependencies
  pip install --no-cache-dir numpy && \
  apt-get install -y python3-opencv && \
  pip install --no-cache-dir networkx==3.1 && \
  pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 && \
  pip install tqdm && \
  pip install matplotlib && \
  pip install pandas
  
# Set environment variables.
ENV HOME /root

# Define working directory.
WORKDIR /root

# Define default command.
CMD ["bash"]
