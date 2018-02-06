import sparknlp.internal as _internal


def get_config_path():
    return _internal._ConfigLoaderGetter().apply()

def set_config_path(path):
    _internal._ConfigLoaderSetter(path).apply()
