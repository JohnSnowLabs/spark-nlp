package com.johnsnowlabs.nlp.annotators.parser.typdep.util

object DictionaryTypes extends Enumeration {
  type DictionaryTypes = Value
  val POS, WORD, DEP_LABEL, WORD_VEC, TYPE_END = Value
}
