import sys
import sparknlp.annotator
from sparknlp.base import DocumentAssembler, Finisher

sys.modules['com.jsl.nlp.annotators'] = annotator
sys.modules['com.jsl.nlp.annotators.ner'] = annotator
sys.modules['com.jsl.nlp.annotators.ner.regex'] = annotator
sys.modules['com.jsl.nlp.annotators.ner.crf'] = annotator
sys.modules['com.jsl.nlp.annotators.pos'] = annotator
sys.modules['com.jsl.nlp.annotators.pos.perceptron'] = annotator
sys.modules['com.jsl.nlp.annotators.sbd'] = annotator
sys.modules['com.jsl.nlp.annotators.sbd.pragmatic'] = annotator
sys.modules['com.jsl.nlp.annotators.sda'] = annotator
sys.modules['com.jsl.nlp.annotators.sda.pragmatic'] = annotator
sys.modules['com.jsl.nlp.annotators.sda.vivekn'] = annotator
sys.modules['com.jsl.nlp.annotators.spell'] = annotator
sys.modules['com.jsl.nlp.annotators.spell.norvig'] = annotator

annotators = annotator
