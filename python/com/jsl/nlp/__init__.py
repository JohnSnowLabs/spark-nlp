import sys

if sys.version_info[0] == 2:
    import sparknlp.annotator
    sys.modules['com.jsl.nlp'] = sparknlp.annotator
else:
    import sparknlp
    sys.modules['com.jsl.nlp'] = sparknlp
