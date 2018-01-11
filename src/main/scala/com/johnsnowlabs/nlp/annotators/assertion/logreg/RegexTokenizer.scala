package com.johnsnowlabs.nlp.annotators.assertion.logreg

/**
  * Created by jose on 18/12/17.
  */
class RegexTokenizer extends Tokenizer{

  /* these match the behavior we had when tokenizing sentences for word embeddings */
  val punctuation = Seq(".", ":", ";", ",", "?", "!", "+", "-", "_", "(", ")", "{",
    "}", "#", "mg/kg", "ml", "m2", "cm", "/", "\\", "\"", "'", "[", "]", "%", "<", ">", "&", "=")

  val percent_regex = """([0-9]{1,2}\.[0-9]{1,2}%|[0-9]{1,3}%)""".r
  val number_regex = """([0-9]{1,6})""".r

  override def tokenize(sent: String): Array[String] = {
    // replace percentage
    var tmp = percent_regex.replaceAllIn(sent, " percentnum ")

    // unbind special chars
    for (c <- punctuation) {
      tmp = tmp.replaceAllLiterally(c, " " + c + " ")
    }

    // replace any num
    val result = number_regex.replaceAllIn(tmp, " digitnum ").toLowerCase.split(" ").filter(_!="")
    result
  }

}
