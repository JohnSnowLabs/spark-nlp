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

  def searchLanguage(text: String) = {
    // tokenize and search token in keys
    val tokens = text.split(" ").map(_.toLowerCase).zipWithIndex.toMap
    val months = dictionary.getOrElse("month", Map.empty[String, Any])
    val matches = tokens filterKeys months.asInstanceOf[Map[String, Any]].keySet
    println(matches)
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
