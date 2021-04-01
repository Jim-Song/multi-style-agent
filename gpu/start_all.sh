#!/bin/bash
if [ -d /dockerdata ]; then
    mv /data1/* /dockerdata
    rm -r /data1
    ln -sf /dockerdata /data1
fi

# wait for all gpu ready
python parse_iplist.py gpu.iplist gpu.iplist.num

# check env
cd /data1 && bash check_env.sh

# set default mem_pool_num 
cd /data1 && sed -i "s/mem_pool_num=.*/mem_pool_num=$CARD_NUM/g" ./config.conf

# set variable
cd /data1 && source ./config.conf


# modify common.conf & run_multi.sh
gpu_iplist=/data1/gpu.iplist
cd /data1/reinforcement_platform/rl_learner_platform/tool/ && bash set_gpu.sh $gpu_iplist $mem_pool_num


#-----------start process----------------#
# start monitor
cd / && bash start_monitor.sh
cd /data1/monitor_agent/op && bash start.sh
#cd /data1/avg_dbwriter/op && bash start.sh

# start mem_pool if use
if [[ $use_mem_pool == 1 ]];then
    cd /data1/reinforcement_platform/mem_pool_server_pkg
    end_port=$[35200 + $mem_pool_num - 1]
    bash clean_all.sh 35200 $end_port && rm -rf mem_pool_server_p352*
    bash setup_pkg.sh 35200 $end_port
    bash start_all.sh 35200 $end_port > start.log
    echo "start mem pool, wait..."
    sleep 15
fi


echo "Start RL learner"
cd /data1/reinforcement_platform/rl_learner_platform/
bash kill.sh && bash clean.sh

if [[ ${IS_MASTER} == "1" ]];then
    # start training
    cd /data1/reinforcement_platform/rl_learner_platform/
    nohup bash start.sh > start.log 2>&1 &

    # start check & send checkpoint
    cd /data1/reinforcement_platform/send_model/
    nohup bash start_check_and_send_checkpoint.sh > send.log 2>&1 &

    # start model pool
    cd /data1/reinforcement_platform/model_pool && nohup python run_model_pool.py --roles=server --model_save_path=/data1/reinforcement_platform/send_model/model --ports=10013:10014 --is_del_model=False > gpu_model_pool.log 2>&1 &

    # start save model
    cd /data1/reinforcement_platform/send_model && nohup sh realtime_save_model.sh $TASK_ID /data1/reinforcement_platform/rl_learner_platform/ 3600 > realtime_save_model.log 2>&1 &
fi
