import sparknlp.internal as _internal


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()

def ExternalResource(path, read_as, options):
    return _internal._ExternalResource(path, read_as, options).apply()

