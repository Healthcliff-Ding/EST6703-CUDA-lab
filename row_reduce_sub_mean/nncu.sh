#!/bin/sh

ncu -k row_reduce_sub_mean0_kernel -c 2 \
    --set detailed \
    --section MemoryWorkloadAnalysis_Tables \
    --section SchedulerStats \
    --section WarpStateStats \
    -o "row_reduce_sub_mean0_kernel$1" \
    a.out 