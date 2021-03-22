#include "sample_parse.h"
using namespace sample_parse;

bool SampleParse::toSample(char* origin_data, int data_len, char* sample_data, std::string mask, int task_id, std::string task_uuid) {
   if(p_data.ParseFromArray(origin_data, data_len) == false)
       return false;
   int offset = 0;
   // add feature
   std::memcpy(sample_data + offset, p_data.mutable_feature()->data(), p_data.mutable_feature()->size());
   offset += p_data.mutable_feature()->size();
   // add advantage
   float advantage = p_data.advantage();
   std::memcpy(sample_data + offset, (char*)&advantage, sizeof(advantage));
   offset += sizeof(advantage);
   //add action_list
   int action_list_len = p_data.action_list_size();
   for (int i = 0; i < action_list_len; i++) {
       float action = p_data.action_list(i);//int32-->float
       std::memcpy(sample_data + offset + i*sizeof(float), (char*)&action, sizeof(float));
   }
   offset += action_list_len * sizeof(float);
   //add neg_log_pis
   float neg_log_pis = p_data.neg_log_pis();
   std::memcpy(sample_data + offset, (char*)&neg_log_pis, sizeof(neg_log_pis));
   offset += sizeof(neg_log_pis);
   //add value
   float value = p_data.value();
   std::memcpy(sample_data + offset, (char*)&value, sizeof(value));
   offset += sizeof(value); 
   return true;
}

