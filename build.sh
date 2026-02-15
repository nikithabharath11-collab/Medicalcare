#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
python train.py
python -c "from app import app, db; app.app_context().__enter__(); db.create_all(); print('Database tables created.')"
