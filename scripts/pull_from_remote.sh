#!/bin/bash

HOST="iit"
REMOTE_DIR_PATH="/home/darshan/aks_test"

# rsync arguments
RSYNC_OPTS="-avu --progress \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.swp'"

rsync $RSYNC_OPTS $HOST:$REMOTE_DIR_PATH/ ./