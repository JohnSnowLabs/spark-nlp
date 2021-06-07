package com.johnsnowlabs.nlp.annotators

object DateMatcherTranslator extends Serializable {

  val Language = "it2"
  val DictionaryPath = s"src/main/resources/date-matcher/date_translation_data/$Language.json"
//  val DictionaryPath = s"/home/wolliqeonii/workspace/dev/jsl/spark-nlp/src/main/resources/date-matcher/date_translation_data/$Language.json"

  def loadTranslationDictionary() = {

    import org.json4s._
    import org.json4s.jackson.JsonMethods._

    import scala.io.Source

    // reading a file
    val jsonString = Source.fromFile(DictionaryPath).mkString
//    println(jsonString)

    val json = parse(jsonString)

    implicit val formats = DefaultFormats
    val jsonMap = json.extract[Map[String, Any]]

    println(jsonMap)
  }

  val dictionary = loadTranslationDictionary()

  def searchLanguage(text: String) = {

  }

  def detectLanguage(text: String, sourceLanguage: String) = {
    println(s"Ciao, detected language: $sourceLanguage")

    searchLanguage(text)
    text
  }

  def translateLanguage(text: String, sourceLanguage: String, destinationLanguage: String = "en") = {
    println("Detecting...")
    val detected = detectLanguage(text, sourceLanguage)
    println("Translating...")
    detected
  }
}
