# -*- coding:utf-8 -*-


class Config:
    INIT_LEARNING_RATE = 2.5e-4
    END_LEARNING_RATE = 1e-5
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.64976
    EPISODE = 20000000
    GAMMA = 0.9966
    LAMDA = 0.95
    STEPS = 157
    EPOCHES = 4
    MINI_BATCH_NUM = 4
    ENV_NAME = 'BowlingNoFrameskip-v4'
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

    #atari ppo, ac
    DATA_SHAPES = [[28228]]
    DATA_SPLIT_SHAPE = [28224, 1, 1, 1, 1]

    #atari dqn
    DATA_SHAPES_DQN = [[56453]]
    DATA_SPLIT_SHAPE_DQN = [28224, 28224, 1, 1, 1, 1, 1]

    INPUT_DIM = [84, 84, 4]
    LIB_GAMECORE = "games.atari.gamecore"
    LIB_AIPROCESS = "games.atari.aiprocess"
    LIB_ENVIRONMENT = "games.atari.environment"
    LIB_PREDICTOR = "algorithms.ppo.predictor"
    LIB_SAMEPLEMANAGER = "algorithms.ppo.sample_manager"
