# -*- coding:utf-8 -*-


class Config:
    INIT_LEARNING_RATE = 2.5e-4
    END_LEARNING_RATE = 1e-5
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.1
    EPISODE = 20000000
    GAMMA = 0.995
    LAMDA = 0.95
    STEPS = 157
    EPOCHES = 4
    MINI_BATCH_NUM = 4
    ENV_NAME = 'PongNoFrameskip-v4'
    MIN_POLICY = 0.00005
    T = 1
    TASK_NAME = 'test_auto'
    DATA_KEYS = "input_data"
    KEY_TYPES = "tf.float32"
    ACTION_DIM = 6
    SERVER_PORT = 30166
    ACTOR_NUM = 200
    LEARNER_NUM = 2
    EACH_LEARNER_NUM = '2'
    PARAMS_PATH = "/data1/reinforcement_platform/rl_learner_platform/model/update"
    GPU_SERVER_LIST = ""
    UPDATE_PATH = '../model/update'
    INIT_PATH = '../model/init'
    MEM_POOL_PATH = './config/mem_pool.host_list'
    SERVER_IP = ''
    TASK_UUID = "123"
    IS_TRAIN = True
    IS_CHECK = False
    LSTM_STEP = 32
    LSTM_UNIT_SIZE = 128

    #atari ppo, ac
    DATA_SHAPES = [[28228]]
    DATA_SPLIT_SHAPE = [28224, 1, 1, 1, 1, LSTM_UNIT_SIZE, LSTM_UNIT_SIZE, 1]

    for i in range(len(DATA_SPLIT_SHAPE)):
        DATA_SPLIT_SHAPE[i] = DATA_SPLIT_SHAPE[i] * LSTM_STEP
    DATA_SHAPES = [[sum(DATA_SPLIT_SHAPE)]]

    #atari dqn
    DATA_SHAPES_DQN = [[56453]]
    DATA_SPLIT_SHAPE_DQN = [28224, 28224, 1, 1, 1, 1, 1]

    INPUT_DIM = [84, 84, 4]

    LIB_GAMECORE = "games.atari.gamecore"
    LIB_AIPROCESS = "games.atari.aiprocess"
    LIB_ENVIRONMENT = "games.atari.environment"
    LIB_PREDICTOR = "algorithms.ppo.predictor"
    LIB_SAMEPLEMANAGER = "algorithms.ppo.sample_manager"

    # style params
    LONG_TERM_STYLE_INTERVAL = 16
    SHORT_TERM_STYLE_INTERVAL = 1
    CAUSAL_CNN_RECEPTIVE_FIELD = 2
    LONG_TERM_STYLE_WEIGHT = 0.1
    SHORT_TERM_STYLE_WEIGHT = 0.00
    NUM_HIDDEN_STYLES = 5

    HIDDEN_CHANNEL = 64
    STYLE_LOSS_WEIGHT = 0.1
