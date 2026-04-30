#!/bin/bash

nginx

cd /app/backend
export FLASK_DEBUG=0
export PORT=5000

exec gunicorn --bind 127.0.0.1:5000 \
    --workers 1 \
    --timeout 120 \
    --threads 4 \
    app:app
