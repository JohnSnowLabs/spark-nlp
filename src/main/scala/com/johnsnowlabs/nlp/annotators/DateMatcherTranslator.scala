package com.johnsnowlabs.nlp.annotators
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source


object DateMatcherTranslator extends Serializable {

  def loadTranslationDictionary(language: String = "en") = {
    val DictionaryPath = s"src/main/resources/date-matcher/date_translation_data/$language.json"
    val jsonString = Source.fromFile(DictionaryPath).mkString

    val json = parse(jsonString)

    implicit val formats = DefaultFormats
    val jsonMap: Map[String, Any] = json.extract[Map[String, Any]]

    //    println(jsonMap)
    jsonMap
  }

  //  val dictionary: Map[String, Any] = loadTranslationDictionary()

  def listFlattener(any: Any): List[Any] =
    any match {
      case i: List[_] => i
      case _ => List(any)
    }

  def detectLanguage(text: String) = {
    val filePath = "src/main/resources/date-matcher/supported_languages.txt"
    val languages = scala.io.Source.fromFile(filePath, "utf-8").getLines.toList

    val matched =
      languages
        .map(l => searchForLanguageMatches(text, l))
        .filterNot(_._1.equals("NF"))

    val mappedDetected: Map[String, Set[String]] = matched.toMap
    println(mappedDetected)
    val matchesLengths: Map[String, Int] = mappedDetected.map{case(k, v) => (k, v.size)}

    val maxValue = matchesLengths.values.max
    val detected = matchesLengths.filter(_._2 == maxValue).toList

    detected
  }

  private def searchForLanguageMatches(text: String, language: String) = {
    val dictionary: Map[String, Any] = loadTranslationDictionary(language)
    // tokenize and search token in keys
    val tokens = text.split(" ").filterNot(_.size <= 2).map(_.toLowerCase).toSet

    val dictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listFlattener).asInstanceOf[List[String]]
      .toSet

    val res: (String, Set[String]) =
      if(!(tokens intersect dictionaryValues isEmpty))
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse("name", "NA")
          .toString -> (tokens intersect dictionaryValues)
      else
        "NF" -> Set.empty

    res
  }

  def detectSourceLanguage(text: String) = {
    val detected: List[(String, Int)] = detectLanguage(text)

    if(detected.size != 1)
      throw new Exception(s"" +
        s"Detected multiple languages. " +
        s"Please specify which one to use using the setSourceLanguage method in the DateMatcher annotator. " +
        s"Detected: ${detected.map(_._1).mkString(", ")} .")

    detected
  }

  def translateLanguage(text: String,
                        sourceLanguage: String,
                        destinationLanguage: String = "en") = {

    // 1. detect source language if None
    val detectedLanguage =
      if(sourceLanguage.isEmpty)
        detectSourceLanguage(text)
      else
        sourceLanguage

    // 2. apply translation
    println(s"Translating from $sourceLanguage to $destinationLanguage...")
    "translated"
  }
}
