#!/bin/bash

source ~/nas4kan/venv/bin/activate

# Add /app and /app/cases/mnist to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:~/nas4kan/:~/nas4kan/cases"

# Run the Python script
python ~/nas4kan/cases/ice_prediction.py


