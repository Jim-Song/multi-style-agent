#include "sample_parse.h"
using namespace sample_parse;

bool SampleParse::toSample(char* origin_data, int data_len, char* sample_data, std::string mask, int task_id, std::string task_uuid) {
   if(p_data.ParseFromArray(origin_data, data_len) == false)
       return false;
   int offset = 0;
   // add feature
//   std::memcpy(sample_data + offset, p_data.mutable_feature()->data(), p_data.mutable_feature()->size());
//   offset += p_data.mutable_feature()->size();
    for (int i = 0; i < p_data.samples_size(); ++i)
    {
        std::memcpy(sample_data + offset, p_data.mutable_samples(i)->mutable_feature()->data(), p_data.mutable_samples(i)->mutable_feature()->size());
        offset += p_data.mutable_samples(i)->mutable_feature()->size();
    }


   // add advantage
//   float advantage = p_data.advantage();
//   std::memcpy(sample_data + offset, (char*)&advantage, sizeof(advantage));
//   offset += sizeof(advantage);
   // add advantage
    for (int i = 0; i < p_data.samples_size(); ++i)
    {
        float advantage = p_data.mutable_samples(i)->advantage();
        std::memcpy(sample_data + offset, (char*)&advantage, sizeof(advantage));
        offset += sizeof(advantage);
    }

   //add action_list
//   int action_list_len = p_data.action_list_size();
//   for (int i = 0; i < action_list_len; i++) {
//       float action = p_data.action_list(i);//int32-->float
//       std::memcpy(sample_data + offset + i*sizeof(float), (char*)&action, sizeof(float));
//   }
//   offset += action_list_len * sizeof(float);
    int action_list_len = p_data.mutable_samples(0)->action_list_size();
    for (int i = 0; i < p_data.samples_size(); ++i)
    {
        for (int j = 0; j < action_list_len; j++) {
            float action = p_data.mutable_samples(i)->action_list(j);//int32-->float
            std::memcpy(sample_data + offset, (char*)&action, sizeof(float));
            offset += sizeof(float);
        }
    }


   //add neg_log_pis
//   float neg_log_pis = p_data.neg_log_pis();
//   std::memcpy(sample_data + offset, (char*)&neg_log_pis, sizeof(neg_log_pis));
//   offset += sizeof(neg_log_pis);
   //add neg_log_pis
    for (int i = 0; i < p_data.samples_size(); ++i)
    {
        float action_neg_log_probs = p_data.mutable_samples(i)->neg_log_pis();
        std::memcpy(sample_data + offset, (char*)&action_neg_log_probs, sizeof(float));
        offset += sizeof(float);
    }

   //add value
//   float value = p_data.value();
//   std::memcpy(sample_data + offset, (char*)&value, sizeof(value));
//   offset += sizeof(value);
   //add value
    for (int i = 0; i < p_data.samples_size(); ++i)
    {
        float value = p_data.mutable_samples(i)->value();
        std::memcpy(sample_data + offset, (char*)&value, sizeof(value));
        offset += sizeof(float);
    }

   //add lstm_c
   for (int i = 0; i < p_data.samples_size(); ++i)
    {
       std::memcpy(sample_data + offset, p_data.mutable_samples(i)->mutable_lstm_c()->data(), p_data.mutable_samples(i)->mutable_lstm_c()->size());
       offset += p_data.mutable_samples(i)->mutable_lstm_c()->size();
    }
   //add lstm_h
   for (int i = 0; i < p_data.samples_size(); ++i)
    {
       std::memcpy(sample_data + offset, p_data.mutable_samples(i)->mutable_lstm_h()->data(), p_data.mutable_samples(i)->mutable_lstm_h()->size());
       offset += p_data.mutable_samples(i)->mutable_lstm_h()->size();
    }
    //add hidden_style
   for (int i = 0; i < p_data.samples_size(); ++i)
    {
        float style = p_data.mutable_samples(i)->style();
        std::memcpy(sample_data + offset, (char*)&style, sizeof(style));
        offset += sizeof(float);
    }

   return true;
}

