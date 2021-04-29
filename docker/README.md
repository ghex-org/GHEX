# How to build and run container

## How to build

```
cd /path/to/GHEX/docker/cpu
docker build --build-arg GCC_VERSION=9 -t mbianco/ghex-cpu:gcc-9 .

cd /path/to/GHEX/docker/gpu_amd
docker build --build-arg REPOSITORY=mbianco/ghex-cpu --build-arg TAG=gcc-9 -t mbianco/ghex-gpu-amd:gcc-9 .

cd /path/to/GHEX/docker/gpu_nvidia
docker build --build-arg REPOSITORY=mbianco/ghex-cpu --build-arg TAG=gcc-9 -t mbianco/ghex-gpu-nvidia:gcc-9 .

```

## How to run

```
docker run -v /path/to/GHEX/:/mnt -it --entrypoint /bin/bash REPOSITORY:TAG
```
