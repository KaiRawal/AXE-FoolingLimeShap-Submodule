#!/bin/bash
echo "============================================================================="


echo $EVAL_PGI_STD

# Run the COMPAS experiment
echo "Running COMPAS experiment..."
python3 compas_experiment.py

echo " - - - - - - - - - - - - - - - - - - "

# Run the Communities and Crime experiment
echo "Running Communities and Crime experiment..."
python3 cc_experiment.py

echo " - - - - - - - - - - - - - - - - - - "

echo "Running German experiment..."
python3 german_experiment.py

echo "============================================================================="

python3 german_experiment.py
python3 compas_experiment.py
python3 cc_experiment.py


echo "============================================================================="

