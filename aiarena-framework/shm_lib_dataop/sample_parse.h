#ifndef _SAMPLE_PARSE_H
#define _SAMPLE_PARSE_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <fstream>
#include <unistd.h>
#include <exception>
#include <sstream>
#include <iostream>
#include <cstring>
#include <string>
#include "proto/gym_rl.pb.h"

namespace sample_parse {
    class SampleParse {
     private:
         gym_rl::RlFragmentInfo p_data;
     public:
         bool toSample(char* origin_data, int data_len, char* sample_data, std::string mask, int task_id, std::string task_uuid);
    };
}
#endif
