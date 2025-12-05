#!/bin/bash

# Start timer
start_time=$(date +%s)

echo "Starting all scripts..."

# Run each script on a different GPU
CUDA_VISIBLE_DEVICES=1 python feature_script.py &
PID1=$!

CUDA_VISIBLE_DEVICES=2 python QBC_script.py &
PID2=$!

CUDA_VISIBLE_DEVICES=3 python dist_script.py &
PID3=$!

# Wait for all scripts to complete
wait $PID1
wait $PID2
wait $PID3

# End timer and compute runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "Total runtime: ${runtime} seconds" | tee runtime.txt

echo "Done."