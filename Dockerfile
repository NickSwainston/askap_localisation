FROM continuumio/miniconda3

COPY conda.yml .
RUN conda env update -n root -f conda.yml && \
    conda clean -a

WORKDIR /tmp/vcstools_build
RUN git clone https://github.com/CIRA-Pulsars-and-Transients-Group/vcstools.git && \
    cd vcstools && \
    python setup.py install
    #pip install mwa_vcstools

COPY askap_beam_localisation.py /usr/local/bin