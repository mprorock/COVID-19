# Dockerized environment for running Jupyter Lab on a mounted filesystem.
#
# Building:
#
#   docker build -t covid19 .
#
# Running
#
#   docker run --init --rm -it -v $(pwd):/srv/data -p:8888:8888 covid19
#
# Accessing
#
#   Running the Docker container will produce output similar to the following,
#   click on the URL for 127.0.0.1 to access the local notebook server:
#
#   ```
#   To access the notebook, open this file in a browser:
#       file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
#   Or copy and paste one of these URLs:
#       http://4cf07c9612ca:8888/?token=fb2428b60c048d78c11c37505ec681ce06033e89043f5731
#    or http://127.0.0.1:8888/?token=fb2428b60c048d78c11c37505ec681ce06033e89043f5731
#   ```

FROM ubuntu:18.04 as builder

WORKDIR /build

# Run debian commands in non-interactive mode
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NONINTERACTIVE_SEEN="true"

# Update system and install basics
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y wget

# conda runs with /bin/sh, which is dash on ubuntu. Some of the python package
# activate/deactivate scripts are bash scripts with non-posix compliant code
# that fails under dash with errors like 'Syntax error: "(" unexpected'.
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN dpkg-reconfigure dash

# Install Miniconda
ARG SHA256_HASH="957d2f0f0701c3d1335e3b39f235d197837ad69a944fa6f5d8ad2c686b69df3b"
ARG SCRIPT_NAME="Miniconda3-latest-Linux-x86_64.sh"
RUN wget -q "https://repo.anaconda.com/miniconda/${SCRIPT_NAME}"
RUN echo "${SHA256_HASH} ${SCRIPT_NAME}" | sha256sum --check --status
RUN /bin/bash ${SCRIPT_NAME} -b -p /opt/miniconda

# Set up environment and install necessary packages, run as one chain so that
# environment is in place for `conda install`
RUN . /opt/miniconda/bin/activate             && \
    conda init bash                           && \
    conda update -y -n base -c defaults conda && \
    conda install -y -c conda-forge              \
        scikit-learn                             \
        plotly                                   \
        seaborn                                  \
        pandas                                   \
        numpy                                    \
        matplotlib                               \
        ipywidgets                               \
        fbprophet                                \
        jupyterlab

FROM ubuntu:18.04

# Run debian commands in non-interactive mode
ARG DEBIAN_FRONTEND="noninteractive"
ARG DEBCONF_NONINTERACTIVE_SEEN="true"

# Repeat system update and dash reconfiguration on final image
RUN apt-get update && apt-get upgrade -y
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN dpkg-reconfigure dash

# Copy miniconda environment
COPY --from=builder /opt/miniconda /opt/miniconda

EXPOSE 8888/tcp
VOLUME /srv/data

RUN apt-get install -y nodejs
ENTRYPOINT [ "/opt/miniconda/bin/jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--allow-root", "/srv/data" ]