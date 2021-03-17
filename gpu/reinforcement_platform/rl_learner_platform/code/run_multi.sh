#!/bin/bash
Nodelist=cc2d23ab818f4142ae99511ab2c9fcc4-learner-0.aiarena-shenzhen-1.svc.cluster.local:1
Num_process=1

net_card_name=eth1
result=`cat /etc/os-release | grep Ubuntu`
if [[ "$result" != "" ]]
then
    net_card_name=eth0
    echo "Ubuntu system, net_card_name is" $net_card_name
else
    echo "tlinux system, net_card_name is" $net_card_name
fi

work_path=$(pwd)/code/
mpirun --allow-run-as-root -np ${Num_process} -H ${Nodelist} -bind-to none -map-by slot \
--mca btl_openib_want_cuda_gdr 1 -mca coll_fca_enable 0 \
--report-bindings --display-map --mca btl_openib_rroce_enable 1 --mca pml ob1 --mca btl ^openib \
--mca btl_openib_cpc_include rdmacm  --mca coll_hcoll_enable 0  --mca plm_rsh_no_tree_spawn 1 \
--mca orte_keep_fqdn_hostnames t \
-x NCCL_IB_DISABLE=1 \
-x NCCL_SOCKET_IFNAME=$net_card_name \
-x NCCL_DEBUG=INFO -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_HCA=mlx5_2:1,mlx5_3:1 -x NCCL_IB_SL=3 -x NCCL_NET_GDR_READ=1 \
-x NCCL_CHECK_DISABLE=1  -x NCCL_LL_THRESHOLD=16384 -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 -x HOROVOD_FUSION_THRESHOLD=1 -x HOROVOD_CYCLE_TIME=0.5  -x NCCL_DEBUG=INFO \
-x LD_LIBRARY_PATH  python ${work_path}/train.py \
--variable_update horovod \
--custom_dataformat True > ./log/trace1.log 2>&1
