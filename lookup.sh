#!/bin/bash

for iter in {1..5}
do
    ./run_carla.sh &
    python3 create_risk_lookup.py --file_id=$iter

done
