cat > start.sh <<'EOF'
#!/usr/bin/env bash
set -e

MODEL_DIR=/app/models
mkdir -p "$MODEL_DIR"

# Download model if not present (replace URL)
if [ ! -f "$MODEL_DIR/model.pt" ]; then
  echo "Downloading model..."
  wget --timeout=60 -O "$MODEL_DIR/model.pt" "https://huggingface.co/<user>/<repo>/resolve/main/model.pt"
fi

# Launch app via wrapper (app.py)
exec python app.py
EOF

chmod +x start.sh
git add start.sh
git commit -m "Add start.sh to download model at container start"
