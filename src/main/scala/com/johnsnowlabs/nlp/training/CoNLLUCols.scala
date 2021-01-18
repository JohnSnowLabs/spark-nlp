package com.johnsnowlabs.nlp.training

object CoNLLUCols extends Enumeration {
  /** CoNLL-U columns [[https://universaldependencies.org/format.html CoNLL-U format]]
   *
   **/
  type Format
  val ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = Value
}
