#!/bin/bash

if [ $# -lt 3 ];then
echo "usage $0 training_type actor_num game_name(atari/kinghonour)"
exit
fi

actor_type=$1
actor_num=$2
game_name=$3

if [ ! -d "./log" ]; then
    mkdir log
else
    rm -r ./log/ && mkdir log
fi

if [ -f "*.log" ]; then
    rm *.log
fi 

nohup python ./tool/tar_ckpt.py > ./log/tar.log 2>&1 &
nohup bash ./model/delete.sh > ./log/del.log 2>&1 &

cd code;

let actor_num=$actor_num-1
for i in $(seq 0 $actor_num); do
    nohup python actor.py $actor_type $i >> ../log/actor${i}.log 2>&1 &
    sleep 60
done;
