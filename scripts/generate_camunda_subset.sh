#!/bin/bash

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset_name> <model1> [model2 ...]"
    exit 1
fi

DATASET_NAME="$1"
shift

SRC="../data/models/camunda/bpmn-for-research-master/BPMN for Research/English"
DST="../data/models/camunda_${DATASET_NAME}/bpmn-for-research-master/BPMN for Research/English"

mkdir -p "$DST"

MODELS=("$@")

for model in "${MODELS[@]}"; do
    echo "Copying $model..."
    cp -R "$SRC/$model" "$DST/"
done

echo "Dataset camunda_${DATASET_NAME} generated."