#!/bin/bash
# conda init bash;
# conda activate /home/abhimanyu/miniconda3/envs/genz_astra_sim/

# which python

SCRIPT_DIR=$(dirname "$(realpath $0)")
BINARY="${SCRIPT_DIR:?}"/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra
# WORKLOAD=/tmp/genz/chakra/et/microAllReduce
WORKLOAD=/tmp/genz/chakra/et_cleaned/collective_traces
SYSTEM="${SCRIPT_DIR:?}"/system.json
NETWORK="${SCRIPT_DIR:?}"/network.yml
MEMORY="${SCRIPT_DIR:?}"/memory.json

"${BINARY}" \
  --workload-configuration="${WORKLOAD}" \
  --system-configuration="${SYSTEM}" \
  --network-configuration="${NETWORK}"\
  --remote-memory-configuration="${MEMORY}"
