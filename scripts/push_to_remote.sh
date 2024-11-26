#!/bin/bash

HOST="iit"
REMOTE_DIR_PATH="/home/darshan/aks_test"

DOCKER_CONTAINER="humble2"
DOCKER_PATH="~/ros2_ws/src/object_detection"

# list of subdirectories to be copied to remote
SUBDIRS=("marker" "task_alloc" "t3")

# rsync arguments
RSYNC_OPTS="-avu --progress \
    --exclude='*/__pycache__' \
    --exclude=".DS_Store" \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.vscode' \
    --exclude='*.log' \
    --exclude='*.tmp' \
    --exclude='*.swp' \
    --exclude='pull_from_remote.sh' \
    --exclude='push_to_remote.sh'"

# rsync each subdirectory
for subdir in "${SUBDIRS[@]}"; do
    # check if the subdirectory exists if not skip
    if [ ! -d $subdir ]; then
        echo "Skipping $subdir, directory does not exist"
        continue
    fi
    # if subdirectory does not exist on remote, create it
    ssh $HOST "mkdir -p $REMOTE_DIR_PATH/$subdir"
    rsync $RSYNC_OPTS ./$subdir $HOST:$REMOTE_DIR_PATH

    # copy to specified docker container
    # docker cp ./$subdir $DOCKER_CONTAINER:$DOCKER_PATH
done