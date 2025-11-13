#!/usr/bin/env bash
set -e

# ---------------------------------------------------------
# srsRAN 5G Standalone Testbed Installer (Simulation Mode)
# ---------------------------------------------------------
# Tested on Ubuntu 22.04 LTS
# Run: bash srsran_setup/install_srsran.sh
# ---------------------------------------------------------

sudo apt update && sudo apt upgrade -y
sudo apt install -y git build-essential cmake libboost-all-dev \
libsctp-dev libfftw3-dev libmbedtls-dev libzmq3-dev libconfig++-dev \
libssl-dev python3-pip

# Clone srsRAN Project
if [ ! -d "srsRAN_Project" ]; then
  git clone https://github.com/srsran/srsRAN_Project.git
  cd srsRAN_Project
  mkdir build && cd build
  cmake ../
  make -j$(nproc)
  sudo make install
  cd ../..
else
  echo "srsRAN_Project already exists — skipping clone."
fi

# Install Python dependencies
pip3 install -r requirements.txt

echo "✅ srsRAN 5G environment setup complete."
echo "You can now configure and run gNB/UE using the provided config files."
