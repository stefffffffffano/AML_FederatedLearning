#!/usr/bin/env bash

NAME="shakespeare"

cd ../utils

python stats.py --name $NAME

cd ../$NAME