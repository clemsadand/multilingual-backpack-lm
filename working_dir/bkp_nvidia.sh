#!/bin/bash

# Run this to install nvidia-drivers
###
#wget https://us.download.nvidia.com/tesla/550.90.12/nvidia-driver-local-repo-ubuntu2004-550.90.12_1.0-1_amd64.deb
wget https://us.download.nvidia.com/tesla/515.105.01/nvidia-driver-local-repo-ubuntu2004-515.105.01_1.0-1_amd64.deb

#
sudo dpkg -i nvidia-driver-local-repo-ubuntu2004-515.105.01_1.0-1_amd64.deb

#sudo cp /var/nvidia-driver-local-repo-ubuntu2004-550.90.12/nvidia-driver-local-561A55CC-keyring.gpg /usr/share/keyrings/
sudo cp /var/nvidia-driver-local-repo-ubuntu2004-515.105.01/nvidia-driver-local-343E1F4D-keyring.gpg /usr/share/keyrings/

sudo apt-get install -f

sudo apt update
#sudo apt install nvidia-driver-515
sudo apt install nvidia-driver-515

sudo reboot
