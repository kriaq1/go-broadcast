#!/bin/bash

xhost +local:docker
docker run --rm -ti -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix go-broadcast bash
xhost -local:docker
