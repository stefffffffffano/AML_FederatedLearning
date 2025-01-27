#!/usr/bin/env bash

NAME="synthetic"

cd ../utils

python stats.py --name $NAME

cd ../$NAME