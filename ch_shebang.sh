#!/bin/bash

PYTHON_PATH=$(python3 -c "import sys; print(sys.executable)")

if [ $# -eq 0 ]; then
    SCRIPT_PATH=( *.py )
elif [ $# -eq 1 ]; then
    SCRIPT_PATH=( "$1"/*.py )
else
    echo "Usage: $0 [directory]"
    exit 1
fi

for filename in "${SCRIPT_PATH[@]}"; do
    if [ -f "$filename" ]; then
        SHEBANG=$(head -n 1 "$filename")
        if [[ "$SHEBANG" =~ ^#!.*python ]]; then
            tmpfile=$(mktemp)
            tail -n +2 "$filename" > "$tmpfile"
            echo "#!$PYTHON_PATH" > "$filename"
            cat "$tmpfile" >> "$filename"
            rm "$tmpfile"
            echo "Shebang updated to '$PYTHON_PATH' in '$filename'"
        else
            echo "No Python shebang line to update in '$filename'."
        fi
    else
        echo "File does not exist: $filename"
        exit 1
    fi
done