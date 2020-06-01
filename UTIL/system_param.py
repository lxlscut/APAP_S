import yaml


class SystemParam:
    def __init__(self):
        pass

    def get_param(self, key):
        f = open("./config.yaml", 'r', encoding='utf-8')
        cont = f.read()
        config = yaml.safe_load(cont)
        param = config.get(key)
        return param
