#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data1/reinforcement_platform/rl_learner_platform/code/shm_lib"
mount -o size=200M -o nr_inodes=1000000 -o noatime,nodiratime -o remount /dev/shm

nohup bash ./code/run_multi.sh > log/my.log 2>&1 &