FROM continuumio/miniconda2

COPY conda.yml .
RUN conda env update -n root -f conda.yml && \
    conda clean -a

COPY askap_beam_localisation.py /usr/local/bin