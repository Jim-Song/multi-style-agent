# -*- coding:utf-8 -*-


class Config:
    use_zmq = False
    #kinghonour: 1e-5 atari: 2.5e-4
    INIT_LEARNING_RATE = 2.5e-4
    END_LEARNING_RATE = 1e-5
    ALPHA = 0.5
    BETA = 0.01
    EPSILON = 1e-5
    INIT_CLIP_PARAM = 0.1
    # kinghonour:4096 atari:256
    BATCH_SIZE = 128
    EPISODE = 20000000
    GAMMA = 0.995
    LAMDA = 0.95
    STEPS = 128
    EPOCHES = 4
    MINI_BATCH_NUM = 4
    ENV_NAME = 'PongNoFrameskip-v4'
    MIN_POLICY = 0.00005
    T = 1
    TASK_NAME = 'test_auto'
    MEM_PROCESS_NUM = 8
    DATA_KEYS = "input_data"
    KEY_TYPES = "tf.float32"
    ACTION_DIM = 6
    SERVER_PORT = 30166
    ACTOR_NUM = 7
    LEARNER_NUM = 1
    EACH_LEARNER_NUM = 1
    PARAMS_PATH = "/data1/reinforcement_platform/rl_learner_platform/model/update"
    GPU_SERVER_LIST = "127.0.0.1:35911"
    UPDATE_PATH = '../model/update'
    INIT_PATH = '../model/init'
    MEM_POOL_PATH = './config/mem_pool.host_list'
    SERVER_IP = '127.0.0.1'
    TASK_UUID = "123"
    IS_TRAIN = True
    IS_CHECK = False
    LSTM_STEP = 16
    LSTM_UNIT_SIZE = 128

    #atari ppo
    ALGORITHM = "ppo"
    DATA_SHAPES = [[28228]] 
    DATA_SPLIT_SHAPE = [28224, 1, 1, 1, 1]
    INPUT_DIM = [84, 84, 4]
    LIB_GAMECORE = "games.atari.gamecore"
    LIB_AIPROCESS = "games.atari.aiprocess"
    LIB_ENVIRONMENT = "games.atari.environment"
    LIB_PREDICTOR = "algorithms.ppo.predictor"
    LIB_SAMEPLEMANAGER = "algorithms.ppo.sample_manager"

    #atari ac
    #ALGORITHM = "ac"
    #DATA_SHAPES = [[28228]] 
    #DATA_SPLIT_SHAPE = [28224, 1, 1, 1, 1]
    #INPUT_DIM = [84, 84, 4] 
    #LIB_GAMECORE = "games.atari.gamecore"
    #LIB_AIPROCESS = "games.atari.aiprocess"
    #LIB_ENVIRONMENT = "games.atari.environment"
    #LIB_PREDICTOR = "algorithms.ac.predictor"
    #LIB_SAMEPLEMANAGER = "algorithms.ac.sample_manager"
    
    #atari dqn
    #ALGORITHM = "dqn"
    #DATA_SHAPES = [[46453]]
    #DATA_SPLIT_SHAPE = [28224, 28224, 1, 1, 1, 1, 1]
    #INPUT_DIM = [84, 84, 4]
    #LIB_GAMECORE = "games.atari.gamecore"
    #LIB_AIPROCESS = "games.atari.aiprocess"
    #LIB_ENVIRONMENT = "games.atari.environment"
    #LIB_PREDICTOR = "algorithms.dqn.predictor"
    #LIB_SAMEPLEMANAGER = "algorithms.dqn.sample_manager"
