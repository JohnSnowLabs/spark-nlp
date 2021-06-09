package com.johnsnowlabs.nlp.annotators
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.io.Source


object DateMatcherTranslator extends Serializable {

  def loadDictionary(language: String = "en") = {
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
    val dictionary: Map[String, Any] = loadDictionary(language)
    // tokenize and search token in keys
    val tokens = text.split(" ").filterNot(_.size <= 2).map(_.toLowerCase).toSet // FIXME more than 2 chars?

    val dictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listFlattener).asInstanceOf[List[String]]
      .toSet

    val intersection = tokens intersect dictionaryValues

    val matchingLanguages: (String, Set[String]) =
      if(!intersection.isEmpty)
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse("name", "NA")
          .toString -> intersection
      else
        "NF" -> Set.empty

    matchingLanguages
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

  type DateMatcherIndexedToken = (String, Int)

  def matchIndexedToken: (DateMatcherIndexedToken, Map[String, Any]) => DateMatcherIndexedToken =
    (indexToken, dictionary) => {
      println(indexToken)
      val (token, index) = indexToken
      val keys = dictionary.keySet

      val getListifiedValues: String => List[String] = dictionary.getOrElse(_, "NA") match {
        case l: List[String] => l
        case s: String => List(s)
        case m: Map[String, Any] => m.keySet.toList
      }

      val translated: Set[(String, Int)] =
        for(k <- keys
            if getListifiedValues(k).contains(token.toLowerCase)
            ) yield (k, index)

      println(translated)

      if(!translated.isEmpty)
        translated.head
      else
        indexToken
    }

  def applyTranslation(translatedIndexedToken: Array[(String, Int)], text: String) = {
    val tokens = text.split(" ")
    translatedIndexedToken.map(t => tokens(t._2) = t._1)
    tokens.mkString(" ")
  }

  def translateToDestinationLanguage(text: String, sourceLanguage: String, destinationLanguage: String = "en") = {
    val sourceLanguageDictionary: Map[String, Any] = loadDictionary(sourceLanguage)

    val translatedIndexedToken: Array[DateMatcherIndexedToken] =
      text.split(" ").zipWithIndex.map(matchIndexedToken(_, sourceLanguageDictionary))

    applyTranslation(translatedIndexedToken, text)
  }

  def translateLanguage(text: String,
                        sourceLanguage: String,
                        destinationLanguage: String = "en") = {

    // 1. detect source language
    val detectedSourceLanguage =
      if(sourceLanguage.isEmpty)
        detectSourceLanguage(text).toString
      else
        sourceLanguage

    // 2. apply translation
    println(s"Translating from $detectedSourceLanguage to $destinationLanguage...")

    val translated = translateToDestinationLanguage(text, detectedSourceLanguage, destinationLanguage)

    translated
  }
}
