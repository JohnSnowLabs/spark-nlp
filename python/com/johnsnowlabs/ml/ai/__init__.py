import sys

if sys.version_info[0] == 2:
    raise ImportError(
        "Spark NLP for Python 2.x is deprecated since version >= 4.0. "
        "Please use an older versions to use it with this Python version."
    )
else:
    import sparknlp
    sys.modules['com.johnsnowlabs.ml.ai'] = sparknlp