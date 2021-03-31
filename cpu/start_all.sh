#!/bin/bash

if [ ! $CARD_NUM ]; then
  echo "CARD_NUM is NULL!" 
  exit
fi

if [ ! $MODEL_POOL_LIST ]; then
  echo "MODEL_POOL_LIST is NULL!" 
  exit
fi


#cd /reinforcement_platform/actor_platform; mv code/ code_bak/; tar -xvzf code.tar.gz

# start monitor
cd /monitor_agent/op/ && bash start.sh

# set env
cd /reinforcement_platform
sed -i "s/mem_pool_num=.*/mem_pool_num=$CARD_NUM/g" ./config.conf
. ./config.conf
touch cpu.iplist
gpu_iplist=./gpu.iplist
cpu_iplist=./cpu.iplist
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/reinforcement_platform"
cpu_actor_num=$CPU_NUM

bash setup_param.sh $gpu_iplist $cpu_iplist $training_type $steps $env_name $task_name $action_dim $game_name $mem_pool_num 
# Setup and start actor
cd /reinforcement_platform/actor_platform; bash kill.sh

cd /reinforcement_platform/actor_platform/model; rm -rf update; mkdir update

cd /reinforcement_platform/model_pool && nohup python -u run_model_pool.py --roles=client --remote_model_pool_addrs=${MODEL_POOL_LIST} --model_save_path=/reinforcement_platform/actor_platform/model/update > client.log 2>&1 &

cd /reinforcement_platform/actor_platform; nohup bash run_actor.sh $training_type $cpu_actor_num $game_name > run_actor.log 2>&1 &
