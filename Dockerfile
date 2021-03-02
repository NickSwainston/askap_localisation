FROM continuumio/miniconda3

COPY conda.yml .
RUN conda env update -n root -f conda.yml && \
    conda clean -a

RUN pip install mwa_vcstools==2.4

COPY askap_beam_localisation.py /usr/local/bin