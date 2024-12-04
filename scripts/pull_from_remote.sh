#!/bin/bash

HOST="iit2"
REMOTE_DIR_PATH="/home/ps2-mid/ws"

SUBDIRS=("vision-exp" "images")

# rsync arguments
RSYNC_OPTS="-avu --progress \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='*.pt' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*/.ipynb_checkpoints' \
    --exclude='.chromadb/' \
    --exclude='*.swp'"

# rsync each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    # check if the subdirectory exists if not skip
    if [ ! -d $subdir ]; then
        echo "Skipping $subdir, directory does not exist"
        continue
    fi
    # if subdirectory does not exist on remote, create it
    mkdir -p $subdir
    rsync $RSYNC_OPTS $HOST:$REMOTE_DIR_PATH/$subdir/ ./$subdir/
done
