pip() {
    if [ "$1" == "install" ]; then
        command uv pip install --system "${@:2}"
    else
        command pip "$@"
    fi
}