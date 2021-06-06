package com.johnsnowlabs.nlp.annotators

import scala.util.parsing.json.JSON

object DateMatcherTranslator extends Serializable {

  val DictionaryPath = "src/main/resources/date-matcher/date_translation_data/it.json"

//  val dictionary = SparkSession.builder().getOrCreate()
//    .read
//    .option("multiLine", true).option("mode", "PERMISSIVE")
//    .json(DictionaryPath)
  val dictionary = JSON.parseFull(DictionaryPath)

  def detectLanguage(text: String) = {
    println(s"Ciao, detected language: $text")
    text
  }

  def translateLanguage(text: String, sourceLanguage: String, destinationLanguage: String = "en") = {
    // search language and translate
    ""
  }
}
