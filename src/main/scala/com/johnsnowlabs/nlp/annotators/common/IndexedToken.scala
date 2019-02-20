package com.johnsnowlabs.nlp.annotators.common

case class IndexedToken(token: String, begin: Int = 0, end: Int = 0, sentenceid:Option[String] = None)
