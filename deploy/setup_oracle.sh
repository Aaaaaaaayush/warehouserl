#!/bin/bash
# ==============================================================================
# WarehouseRL — One-Click Oracle Cloud Ubuntu 22.04 LTS ARM Deployment Script
# ==============================================================================

set -e

echo "============================================================"
echo "  Deploying WarehouseRL to Oracle Cloud Ubuntu ARM VM"
echo "============================================================"

# 1. Update system packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose git curl certbot python3-certbot-nginx iptables-persistent

# 2. Enable & start Docker service
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# 3. Configure iptables to open ports 80 and 443 on Oracle Cloud Ubuntu
echo "Opening HTTP (80) and HTTPS (443) firewall ports..."
sudo iptables -I INPUT 6 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save

# 4. Clone / Pull repository
if [ ! -d "warehouserl" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Aayush-TUM/warehouserl.git warehouserl
    cd warehouserl
else
    echo "Pulling latest changes..."
    cd warehouserl
    git pull origin main
fi

# 5. Build & Launch Docker containers
echo "Building and launching containers via docker-compose..."
cd deploy
sudo docker-compose down
sudo docker-compose up --build -d

echo "============================================================"
echo "  [OK] WarehouseRL successfully deployed!"
echo "  Live URL: http://aayush-warehouserl.duckdns.org"
echo "============================================================"
