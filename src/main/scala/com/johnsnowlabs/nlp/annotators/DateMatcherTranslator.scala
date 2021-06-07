package com.johnsnowlabs.nlp.annotators
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source


object DateMatcherTranslator extends Serializable {

  val Language = "it_new"
  val DictionaryPath = s"src/main/resources/date-matcher/date_translation_data/$Language.json"

  def loadTranslationDictionary() = {
    val jsonString = Source.fromFile(DictionaryPath).mkString

    val json = parse(jsonString)

    implicit val formats = DefaultFormats
    val jsonMap: Map[String, Any] = json.extract[Map[String, Any]]

//    println(jsonMap)
    jsonMap
  }

  val dictionary: Map[String, Any] = loadTranslationDictionary()

  def monthFlatter(any: Any): List[Any] =
    any match {
      case i: List[_] => i
      case _ => List(any)
    }

  def searchLanguage(text: String) = {
    // tokenize and search token in keys
    val tokens = text.split(" ").map(_.toLowerCase).toSet

    val months = dictionary
      .getOrElse("month", Map.empty[String, Any])
      .asInstanceOf[Map[String, Any]]
      .values.flatten(monthFlatter).asInstanceOf[List[String]]
      .toSet

    println(months)

    !(tokens intersect months isEmpty)
  }

  def detectSourceLanguage(text: String) = {
    val _sourceLanguage = searchLanguage(text)
    _sourceLanguage
  }

  def translateLanguage(text: String, sourceLanguage: String, destinationLanguage: String = "en") = {
    // detect source language
    val _sourceLanguage = if(!sourceLanguage.equals("en")) detectSourceLanguage(text) else "en"
    // apply translation
    ""
  }
}
