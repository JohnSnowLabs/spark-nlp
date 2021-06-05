package com.johnsnowlabs.nlp.annotators

import scala.util.parsing.json.JSON

object DateMatcherTranslator extends Serializable {

  val DictionaryPath = "src/main/resources/date-matcher/date_translation_data/it.json"

//  val dictionary = SparkSession.builder().getOrCreate()
//    .read
//    .option("multiLine", true).option("mode", "PERMISSIVE")
//    .json(DictionaryPath)
  val dictionary = JSON.parseFull(DictionaryPath)

  def normalizeDateLanguage(source: String, destination: String = "en") = ???

  def detectLanguage(text: String) = {
    println("Ciao, detect language")
    "detected"
  }

  def translate(text: String): Unit ={
    val source = detectLanguage(text)
    normalizeDateLanguage(source, destination = "en")
  }


}
