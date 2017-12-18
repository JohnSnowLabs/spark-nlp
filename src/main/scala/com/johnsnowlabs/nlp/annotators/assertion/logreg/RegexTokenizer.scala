package com.johnsnowlabs.nlp.annotators.assertion.logreg

/**
  * Created by jose on 18/12/17.
  */
class RegexTokenizer extends Tokenizer{

  /* these match the behavior we had when tokenizing sentences for word embeddings */
  val punctuation = Seq(".", ":", ";", ",", "?", "!", "+", "-", "_", "(", ")", "{",
    "}", "#", "mg/kg", "ml", "m2", "cm", "/", "\\", "\"", "'", "[", "]", "%", "<", ">", "&", "=")

  val percent_regex = """([0-9]{1,2}\.[0-9]{1,2}%|[0-9]{1,3}%)"""
  val number_regex = """([0-9]{1,6})"""

  override def tokenize(sent: String): Array[String] = ???

}
