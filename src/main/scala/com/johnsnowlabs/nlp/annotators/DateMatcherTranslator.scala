package com.johnsnowlabs.nlp.annotators

import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.{FileNotFoundException, IOException}
import scala.io.Source


object DateMatcherTranslator extends Serializable {

  val SupportedLanguagesFilePath = "src/main/resources/date-matcher/supported_languages.txt"
  val TranslationDataBaseDir = "src/main/resources/date-matcher/translation-dictionaries/dynamic/"

  val JsonSuffix = ".json"
  val Encoding = "utf-8"

  val NotFound = "NF"
  val SkipChar = "#"
  val SpaceChar = " "
  val NotAvailable = "NA"
  val NameKey = "name"
  val English = "en"

  /**
   * Load dictionary from supported language repository.
   * @param language the language dictionary to load. Default is English.
   * @return a map containing the language dictionary or throws an exception.
   * */
  def loadDictionary(language: String = English) = {
    val DictionaryPath = s"$TranslationDataBaseDir$language$JsonSuffix"

    var jsonString = "";
    try{
      jsonString = Source.fromFile(DictionaryPath).mkString
    } catch {
      case e: FileNotFoundException => println(s"Couldn't find $language file in repository.")
      case e: IOException => println("Got an IOException!")
    }

    val json = parse(jsonString)

    implicit val formats = DefaultFormats
    val jsonMap: Map[String, Any] = json.extract[Map[String, Any]]

    jsonMap
  }

  // Utility method to parse the json files structure
  def listify(any: Any): List[Any] =
    any match {
      case i: List[_] => i
      case _ => List(any)
    }

  def detectLanguage(text: String) = {
    val supportedLanguages =
      scala.io.Source
        .fromFile(SupportedLanguagesFilePath, Encoding)
        .getLines.toList

    val activeLanguages = supportedLanguages.filterNot(_.startsWith(SkipChar)) // skip char

    val mappedDetected = activeLanguages
      .map(l => searchForLanguageMatches(text, l))
      .filterNot(_._1.equals(NotFound))
      .toMap

    val detected = mappedDetected.size match {
      case 0 => List.empty
      case _ =>
        val matchesLengths: Map[String, Int] = mappedDetected.map{case(k, v) => (k, v.size)}
        val maxValue = matchesLengths.values.max
        matchesLengths.filter(_._2 == maxValue).toList
    }

    detected
  }

  /**
   * Search for language matches token by token.
   * @param text: the text to process for matching.
   * @param language: the 2 characters string identifying a supported language.
   * @return
   * */
  private def searchForLanguageMatches(text: String, language: String) = {
    val dictionary: Map[String, Any] = loadDictionary(language)

    // tokenize and search token in keys
    val tokens = text.split(SpaceChar)
      .map(_.toLowerCase)
      .toSet

    val dictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet

    println(tokens.mkString("|"))
    println(dictionaryValues.mkString("|"))

    // Search matches for each token in retrieved values from dictionary map
    val staticValuesIntersection =
      tokens.intersect(dictionaryValues)
        .filterNot(_.contains("#V#"))

    println(s"staticValuesIntersection: $staticValuesIntersection")

    val matchingLanguages: (String, Set[String]) =
      if(!staticValuesIntersection.isEmpty)
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse(NameKey, NotAvailable)
          .toString -> staticValuesIntersection
      else
        NotFound -> Set.empty

    matchingLanguages
  }

  /**
   * Detect the source language by looking for date language matches.
   * @param text: the text to detect.
   * @return the detected language as tuple of matched languages with their frequency.
   * */
  def detectSourceLanguage(text: String) = {
    val detectedLanguage: List[(String, Int)] = detectLanguage(text)

    println(s"==> detected languages: $detectedLanguage")

    // TODO sort?
    // must be a single element list.
    if(detectedLanguage.size != 1)
      English
    else {
      val (language, _) = detectedLanguage.head
      language
    }

  }

  // utility type
  type DateMatcherIndexedToken = (String, Int)

  /**
   *  Matches the indexed text token against the passed dictionary.
   *  @param indexToken the indexed token to match.
   *  @param dictionary the dictionary to match token against.
   * */
  def matchIndexedToken: (DateMatcherIndexedToken, Map[String, Any]) => DateMatcherIndexedToken =
    (indexToken, dictionary) => {
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

      // only first match if any is returned
      if(!translated.isEmpty)
        translated.head
      else {
        // bypassed as not matched
        indexToken
      }
    }

  /**
   * Apply translation switching token where an index has been translated using the dictionary matching.
   * @param translatedIndexedToken: the translated index token to replace in text.
   * @param text: the original text where translation is applied.
   * @return the text translated using token replacement.
   * */
  def applyTranslation(translatedIndexedToken: Array[(String, Int)], text: String) = {
    val tokens = text.split(SpaceChar)
    translatedIndexedToken.map(t => tokens(t._2) = t._1)
    tokens.mkString(" ")
  }

  /**
   * Translate the text from source language to destination language.
   * @param text the text to translate.
   * @param source the source language.
   * @param destination the destination language.
   * @return the translated text from source language to destination language.
   * */
  private def _translate(text: String, source: String, destination: String = English) = {

    println(s"SOURCE LANG: $source")
    println(s"DESTINATION LANG: $destination")

    if(!source.equals(English)) {
      val sourceLanguageDictionary: Map[String, Any] = loadDictionary(source)

      val translatedIndexedToken: Array[DateMatcherIndexedToken] =
        text
          .split(SpaceChar).zipWithIndex
          .map(matchIndexedToken(_, sourceLanguageDictionary))

      applyTranslation(translatedIndexedToken, text)
    }
    else
      text
  }

  /**
   * Translate the text from source language to destination language.
   * @param text the text to translate.
   * @param source the source language.
   * @param destination the destination language.
   * @return the translated text from source language to destination language.
   * */
  def translate(text: String, source: String, destination: String = English) = {

    // 1. set or detect source language
    val _source =
      source match {
        case s: String if !s.isEmpty && s.length == 2 => s
        case _ => detectSourceLanguage(text)
      }

    // 2. apply translation
    val translated = _translate(text, _source, destination)

    translated
  }
}
