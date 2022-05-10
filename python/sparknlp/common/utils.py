def ExternalResource(path, read_as=ReadAs.TEXT, options={}):
    """Returns a representation fo an External Resource.

    How the resource is read can be set with `read_as`.

    Parameters
    ----------
    path : str
        Path to the resource
    read_as : str, optional
        How to read the resource, by default ReadAs.TEXT
    options : dict, optional
        Options to read the resource, by default {}
    """
    return _internal._ExternalResource(path, read_as, options).apply()

def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()

