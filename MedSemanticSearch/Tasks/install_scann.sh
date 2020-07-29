#!/bin/bash

sudo apt-get update
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt install gcc-9 g++-9
sudo apt-get install libstdc++6
pip install https://storage.googleapis.com/scann/releases/1.0.0/scann-1.0.0-cp37-cp37m-linux_x86_64.whl