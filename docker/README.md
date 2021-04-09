# How to build and run container

## How to build

```
sudo docker build --build-arg REPOSITORY=mbianco/ghex --build-arg GCC_VERSION=9 -t mbianco/ghex:gcc-9 .
```

## How to run

```
sudo docker run -v /path/to/GHEX/:/mnt -it --entrypoint /bin/bash mbianco/ghex:gcc-9
```
