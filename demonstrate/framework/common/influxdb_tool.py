import subprocess
try:
    from influxdb import InfluxDBClient
    INFLUX_INSTALL = True
except Exception as error:
    print("import influxdb error {}".format(error))
    INFLUX_INSTALL = False


class InfluxTool(object):
    def __init__(self, actor_idx, conf_path="/reinforcement_platform/config.conf"):
        cmd = "bash ../tool/get_ip.sh"
        docker_ip = subprocess.check_output(cmd, shell=True).decode().strip()
        cmd = "bash ../tool/get_port.sh"
        docker_port = subprocess.check_output(cmd, shell=True).decode().strip()

        self.actor_idx = int(actor_idx)
        self.docker_addr = docker_ip + ":" + docker_port + ":" + str(actor_idx)
        self.db_conf = {}
        with open(conf_path, 'r') as fin:
            for line in fin.readlines():
                data = line.strip().split("=")
                if len(data) != 2:
                    continue
                if data[0].strip() in ["db_ip", "db_port", "db_name", "db_user",\
                        "db_password", "task_uuid", "task_name"]:
                    self.db_conf[data[0].strip()] = data[1].strip()
        self.is_ok = (len(self.db_conf) == 7)

    def send_rwd_stat(self, reward_details):
        sum_rwd_dict = {}
        sum_rwd_dict["game_len"] = len(reward_details)
        for detail in reward_details:
            for key in detail.keys():
                if key not in sum_rwd_dict:
                    sum_rwd_dict[key] = 0.0
                sum_rwd_dict[key] += detail[key]
        try:
            self.write("reward_info", sum_rwd_dict)
        except Exception as error:
            print("write ifluxdb error {}".format(error))
        return

    def write(self, table_name, data_dict):
        if self.actor_idx % 2 != 0:
            return
        if self.is_ok and INFLUX_INSTALL:
            client = InfluxDBClient(host=self.db_conf["db_ip"], port=self.db_conf["db_port"],\
                    database=self.db_conf["db_name"], username=self.db_conf["db_user"],\
                    password=self.db_conf["db_password"], timeout=1)
            json_body = {}
            json_body["measurement"] = table_name+'_'+self.db_conf['task_name']+'_'+self.db_conf['task_uuid']
            json_body['docker'] = self.docker_addr
            json_body["fields"] = data_dict
            client.write_points([json_body])
            client.close()
        return
