package com.johnsnowlabs.nlp.annotators.common

case class WordWithDependency(word: String, head: Int, dependencyRelation: String, begin: Int, end: Int)
