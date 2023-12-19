#!/bin/sh

ncu -k add_bias0_kernel -c 2 \
    --set detailed \
    --section MemoryWorkloadAnalysis_Tables \
    --section SchedulerStats \
    --section WarpStateStats \
    -o "result$1" \
    a.out 