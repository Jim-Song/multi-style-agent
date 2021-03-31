#!/bin/bash

#ps aux | grep rm_abs | grep -v grep  | awk '{print $2}' | xargs kill -9
ps aux | grep actor.py | grep -v grep | awk '{print $2}' | xargs kill -9
#ps aux | grep server.py | grep -v grep  | awk '{print $2}' | xargs kill -9
ps aux | grep tar_ckpt.py | grep -v grep  | awk '{print $2}' | xargs kill -9
ps aux | grep delete.sh | grep -v grep  | awk '{print $2}' | xargs kill -9
#ps aux | grep test_reward.py | grep -v grep | awk '{print $2}' | xargs kill -9
#ps aux | grep sgame | grep -v grep | awk '{print $2}' | xargs kill -9
