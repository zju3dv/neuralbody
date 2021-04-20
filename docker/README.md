## 1. Build the image

From the root path of the project:
```shell
docker build -f docker/Dockerfile -t neuralbody .
```

You may want to try several times since there are so many packages to be downloaded through the Internet and htpp(s) erros could occur.

## 2. Data preparation

The docker image contains the environment you need to run the project, while you still need to manually download data as described in [INSTALL.md](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md).

Note that the files downloaded are originally tar.gz files, while you need to extract each of them.

An example is like:

```shell
for name in $(ls *.tar.gz); do tar -xvf $name; done
```

## 3. Execution using docker containers


Suppose you are at the root path of the project, run a docker container like:
```shell
docker run -it --rm --gpus=all \
--mount type=bind,source="$(pwd)",target=/app \
--mount type=bind,source=<DATAPATH>,target=/app/data \
neuralbody <COMMAND>
```
where `<COMMAND>` can be obtained from [README.md](https://github.com/zju3dv/neuralbody/blob/master/README.md) and `<DATAPATH>` is your path for data.
