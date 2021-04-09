#__author__ = "ciaochen"
import tensorflow as tf
from atari_ppo.config import Config
from SailLearner.algorithms.base.algorithm import Algorithm as AlgorithmBase


class Algorithm(AlgorithmBase):
    def __init__(self, model):
        super().__init__(model)
        self.init_learning_rate = Config.INIT_LEARNING_RATE 
        self.init_clip_param = Config.INIT_CLIP_PARAM
        self.epsilon = Config.EPSILON
        self.episode = Config.EPISODE
        self.alpha = Config.ALPHA
        self.beta = Config.BETA
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_step = Config.LSTM_STEP
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.style_loss_weight = Config.STYLE_LOSS_WEIGHT

    def build_graph(self, datas, update):
        self.update = tf.cast(update, tf.float32)
        self.learning_rate = tf.maximum(self.init_learning_rate * tf.cast((1 - self.update/self.episode), \
                tf.float32), 5e-5)
        self.clip_range = tf.maximum(self.init_clip_param * tf.cast((1 - self.update/self.episode), tf.float32), 5e-5)
        self._split_data(datas)
        self.model.inference(self.feature, self.lstm_c, self.lstm_h, self.style)
        self._calculate_loss()
        return self.loss, [self.policy_loss, self.value_loss, self.entropy_loss,
                           tf.reduce_mean(self.sampled_reward_sum), self.learning_rate,
                           self.style_loss, tf.reduce_mean(self.long_style_loss),
                           tf.reduce_mean(self.short_style_loss), datas]

    def _split_data(self, datas):
        self.feature, self.sampled_advantage, self.sampled_action, self.sampled_neg_log_pi, self.sampled_value, self.lstm_c_all, self.lstm_h_all, self.style = tf.split(datas, self.data_split_shape, axis=1)
        self.feature = tf.reshape(self.feature, [-1, 28224])
        self.sampled_normalized_advantage = tf.reshape(self.sampled_advantage, [-1])
        self.sampled_action = tf.reshape(self.sampled_action, [-1])
        self.sampled_neg_log_pi = tf.reshape(self.sampled_neg_log_pi, [-1])
        self.sampled_value = tf.reshape(self.sampled_value, [-1])
        self.lstm_c_all = tf.reshape(self.lstm_c_all, [-1, self.lstm_step, self.lstm_unit_size])
        self.lstm_h_all = tf.reshape(self.lstm_h_all, [-1, self.lstm_step, self.lstm_unit_size])
        self.lstm_c = self.lstm_c_all[:, 0, :]
        self.lstm_h = self.lstm_h_all[:, 0, :]
        self.style = tf.cast(tf.reshape(self.style, [-1]), tf.int32)
        self.sampled_reward_sum = self.sampled_value + self.sampled_normalized_advantage
        self.action_rank = self.sampled_action.shape.ndims 
        self.feature.shape.assert_has_rank(self.action_rank + 1)
        self.sampled_normalized_advantage.shape.assert_has_rank(self.action_rank)
        self.sampled_action.shape.assert_has_rank(self.action_rank)
        self.sampled_neg_log_pi.shape.assert_has_rank(self.action_rank)  
        self.sampled_value.shape.assert_has_rank(self.action_rank)
        self.sampled_reward_sum.shape.assert_has_rank(self.action_rank)  

    def _calculate_loss(self):
        self._calc_policy_loss()
        self._calc_value_loss()
        self._calc_entropy_loss()
        self.style_loss = self.model.style_loss
        self.long_style_loss = self.model.long_style_loss
        self.short_style_loss = self.model.short_style_loss
        self.loss = -(self.policy_loss - self.alpha * self.value_loss + self.beta * self.entropy_loss) + \
                    self.style_loss * self.style_loss_weight

    def _calc_policy_loss(self):
        neg_log_pi = self.model.neg_log_prob(self.sampled_action, "neg_log_pi")
        neg_log_pi.shape.assert_has_rank(self.action_rank)
        ratio = tf.exp(self.sampled_neg_log_pi - neg_log_pi, name="ratio")
        clipped_ratio = tf.clip_by_value(ratio, 0.0, 3.0)
        surr1 = clipped_ratio * self.sampled_normalized_advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * \
                self.sampled_normalized_advantage
        self.policy_loss = tf.reduce_mean(tf.minimum(surr1, surr2), name="policy_loss")

    def _calc_value_loss(self):
        value = self.model.value
        value.shape.assert_has_rank(self.action_rank)
        clipped_value = tf.add(self.sampled_value, tf.clip_by_value(value - self.sampled_value,\
                - self.clip_range, self.clip_range), name="clipped_value")
        self.value_loss = tf.multiply(0.5, tf.reduce_mean(tf.maximum(tf.square(value - \
                self.sampled_reward_sum), tf.square(clipped_value - self.sampled_reward_sum))), name="vf_loss")

    def _calc_entropy_loss(self):
        pi_logits = self.model.pi_logits
        logits = pi_logits - tf.reduce_max(pi_logits, axis=-1, keepdims=True)
        exp_logits = tf.exp(logits)
        exp_logits_sum = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        prob = exp_logits / exp_logits_sum
        temp_entropy_loss = tf.reduce_sum(prob * (tf.log(exp_logits_sum) - logits), axis=-1)
        self.entropy_loss = tf.reduce_mean(temp_entropy_loss, name="entropy_loss")

    def get_optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon)
