package com.johnsnowlabs.nlp.annotators.tokenizer.moses

/**
  * Scala Port of the Moses Tokenizer from [[https://github.com/alvations/sacremoses scaremoses]].
  */
private[johnsnowlabs] class MosesTokenizer(lang: String) {
  private final val DEDUPLICATE_SPACE = (raw"""s+""", " ")
  private final val ASCII_JUNK = (raw"""[\000-\037]""", "")
  private final val PAD_NOT_ISALNUM = (raw"""[^[:alnum:]s.'`,-]""", raw""" \1 """) // TODO: consider all other languages

  private def replaceMultidots(text: String): Unit = {
    var processed: String = text
    processed = processed.replaceAll(raw"""\.([\.]+)""", raw""" DOTMULTI\1""")

    while (processed.indexOf("DOTMULTI.") > 0) // re.search(r"DOTMULTI\.", text)
      processed = processed.replaceAll(raw"""DOTMULTI\.([^\.])""", raw"""DOTDOTMULTI \1""")
    processed = processed.replaceAll(raw"""DOTMULTI\.""", raw"""DOTDOTMULTI""")
    text
  }

//  def tokenize(text: String): Unit = {
//    for ((pattern, sub) <- List(DEDUPLICATE_SPACE, ASCII_JUNK)) {
//      text = text.replaceAll(pattern, sub)
//    }
//    text = text.trim()
//
//    text =
//  }
}
