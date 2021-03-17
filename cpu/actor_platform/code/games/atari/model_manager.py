import os
import glob
from framework.common.common_log import CommonLogger
from framework.common.common_func import CommonFunc
from framework.game.model_manager import ModelManager as ModelManager_Base

LOG = CommonLogger.get_logger()


class ModelManager(ModelManager_Base):
    def __init__(self):
        self.version_list = []
        self.key_word = ".completed"
        self.model_path = ""
        self.model_version = ""

    def init_predictor(self, predictor, init_path=None, update_path=None):
        self.predictor = predictor
        self.update_path = update_path
        self.init_path = init_path

    def select_model(self):
        is_load_success = True
        if self._get_model_list():
            self._load_model_version()
        else:
            LOG.error("init_game_model failed")
            is_load_success = False
        return is_load_success

    def process(self, feature):
        return self.predictor.process(feature)

    def get_value(self, feature):
        return self.predictor.get_value(feature)

    def init_model(self,
                   lastest_params=None,
                   config_id=None,
                   model_name=None,
                   mode_type=None):
        LOG.info("set model_path : %s", self.model_path)
        return self.predictor.init_model(self.model_path,
                                         lastest_params,
                                         config_id,
                                         model_name,
                                         mode_type)

    def _get_model_list(self):
        if not os.path.exists(self.update_path):
            LOG.error("path %s did not exist"%(self.update_path))
            return False
        self.version_list = glob.glob(("{}/checkpoints_*" + self.key_word).format(self.update_path))
        self.version_list.sort()
        return True

    def _load_model_version(self):
        #model_path = ""
        #if self.version_list:
        #    model_path = self._get_model_version()
        #else:
        #    model_path = self.init_path + "/model.ckpt"
        #self.model_path = model_path
        while len(self.version_list) == 0:
            self._get_model_list()
        self.model_path = self._get_model_version()

    def _get_model_version(self):
        model_path = ""
        self.model_version = CommonFunc.get_version(self.version_list[len(self.version_list) - 1],
                                                    self.key_word)
        model_path = "%s/%s/model.ckpt"%(self.update_path, self.model_version)
        return model_path
