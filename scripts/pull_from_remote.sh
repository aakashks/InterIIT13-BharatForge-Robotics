#!/bin/bash

HOST="iit2"
REMOTE_DIR_PATH="/home/ps2-mid/ws"

# rsync arguments
RSYNC_OPTS="-avu --progress \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*/.ipynb_checkpoints' \
    --exclude='*/.chromadb/' \
    --exclude='*.swp'"

rsync $RSYNC_OPTS $HOST:$REMOTE_DIR_PATH/ ./