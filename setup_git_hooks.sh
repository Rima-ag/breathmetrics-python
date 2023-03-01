#!/bin/bash

echo $'#!/bin/sh

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" = "main" ]; then
    echo "You can\'t commit directly to main branch"
    exit 1
fi

black .
python lint.py -p ../breathmetrics-python/'> .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit