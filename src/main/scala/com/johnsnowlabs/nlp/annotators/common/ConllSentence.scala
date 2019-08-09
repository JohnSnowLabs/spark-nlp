package com.johnsnowlabs.nlp.annotators.common

case class ConllSentence(dependency: String, lemma: String, uPos: String, xPos: String, deprel: String, head: Int,
                         sentence: Int, begin: Int, end: Int)