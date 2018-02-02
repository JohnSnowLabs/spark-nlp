from sparknlp.common import ConfigLoaderGetter, ConfigLoaderSetter


def get_config_path():
    return ConfigLoaderGetter()()


def set_config_path(path):
    ConfigLoaderSetter(path)()
