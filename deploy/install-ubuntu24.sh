#!/usr/bin/env bash
# Khmer OCR — Ubuntu 24.04 install script. Run as root (sudo bash install-ubuntu24.sh)
set -euo pipefail

APP_DIR=/opt/khmer-ocr

apt update
apt install -y tesseract-ocr tesseract-ocr-khm poppler-utils \
               python3-venv python3-pip nginx fonts-noto git

# Best-quality Khmer model
wget -O /usr/share/tesseract-ocr/5/tessdata/khm.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/khm.traineddata

tesseract --list-langs | grep -q khm && echo "Khmer model OK"

# App (assumes the code is already at $APP_DIR — git clone or rsync it there first)
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
chown -R www-data:www-data "$APP_DIR"

# systemd service
cp deploy/khmer-ocr.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now khmer-ocr

# nginx
cp deploy/nginx-khmer-ocr.conf /etc/nginx/sites-available/khmer-ocr
ln -sf /etc/nginx/sites-available/khmer-ocr /etc/nginx/sites-enabled/khmer-ocr
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

echo "Done. Test: curl localhost/health"
