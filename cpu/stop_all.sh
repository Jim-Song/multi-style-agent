#!/bin/bash
cd /reinforcement_platform/actor_platform; bash kill.sh
ps -ef | grep "run_model_pool" | awk '{print $2}' | xargs kill -9
