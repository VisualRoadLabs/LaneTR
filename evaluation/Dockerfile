# Imagen que compila el evaluador OFICIAL de CULane (C++ + OpenCV) de forma reproducible.
# Build:  docker build -t culane-eval:official -f evaluation/Dockerfile evaluation
# Run:    docker run --rm -v <host_work>:/work culane-eval:official \
#             -a /work/anno -d /work/pred -i /work/anno -l /work/list.txt \
#             -w 30 -t 0.5 -c 1640 -r 590 -f 1 -o /work/out.txt
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make pkg-config libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/culane_eval
COPY culane_official/ ./

RUN make -f Makefile.opencv4 clean && make -f Makefile.opencv4

ENTRYPOINT ["/opt/culane_eval/evaluate"]
