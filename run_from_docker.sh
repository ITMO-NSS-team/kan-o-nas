#!/bin/bash

# Add /app and /app/cases/mnist to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:/app:/app/cases"

# Run the Python script
python /app/cases/ice_prediction.py

