package com.johnsnowlabs.nlp.annotators

import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.{FileNotFoundException, IOException}
import scala.io.Source
import scala.util.matching.Regex


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

    // FIXME delete me
    println(s"------------------------------------------\nLoading dictionary: $DictionaryPath")

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

    println(s"mappedDetected: $mappedDetected")

    mappedDetected
  }

  def stringValuesToRegexes(regexValuesIntersection: Set[String]): Set[Regex] = {
    val res = regexValuesIntersection
      .map(_.replaceAll("#V#", """(\\d+)"""))
      .map(_.toLowerCase)
      .map(s => s.r)
    res
  }

  /**
   * Search for language matches token by token.
   *
   * @param text: the text to process for matching.
   * @param language: the 2 characters string identifying a supported language.
   * @return
   * */
  private def searchForLanguageMatches(text: String, language: String, useTokens: Boolean = true) = {
    val dictionary: Map[String, Any] = loadDictionary(language)

    // contains all the retrieved values
    val dictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet

    val tokenMatches = {
      // tokenize and search token in keys
      val candidateTokens =
        text
          .split(SpaceChar)
          .map(_.toLowerCase)
          .toSet

      //      // FIXME delete me
      //      println(s"Loaded dictionary for language: $language")
      //      println(s"tokens: ${candidateTokens.mkString("|")}")
      //      println(s"dictionaryValues: ${dictionaryValues.mkString("|")}")

      // Search matches for each token in retrieved values from dictionary map
      val tokenMatches =
        candidateTokens
          .intersect(
            dictionaryValues.filterNot(_.contains("#V#")))

      // FIXME delete me
      println(s"tokenMatches: $tokenMatches")

      tokenMatches
    }

    val sentenceMatches =  {
      // TODO add regex processing from strings with placeholders
      val regexValuesIntersection = dictionaryValues.filter(_.contains("#V#")) // if there is a regex

      val regexes = stringValuesToRegexes(regexValuesIntersection)

      val sentenceMatches =
        for(regex <- regexes;
            res = regex.findFirstMatchIn(text)
            if !res.isEmpty && res.size != 0
            ) yield (text, regex)

      // TODO QUI -------------------------------------------------
      println(s"sentenceMatches: ${sentenceMatches.mkString("|")}")
      sentenceMatches
    }

    val globalMatches = tokenMatches.union(sentenceMatches.map(_._2.toString))

    val matchingLanguages: (String, Set[String]) =
      if(globalMatches.size != 0) { // if tokens match

        // FIXME delete me
        println(s"Global matches: ${globalMatches.mkString("=>")}")

        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse(NameKey, NotAvailable)
          .toString -> globalMatches
      } else
        NotFound -> Set.empty

    matchingLanguages
  }

  /**
   * Detect the source language by looking for date language matches.
   * @param text: the text to detect.
   * @return the detected language as tuple of matched languages with their frequency.
   * */
  def detectSourceLanguage(text: String): Map[String, Set[String]] = {
    // e.g. Map(it -> Set((.*) anni fa))
    val detectedLanguage: Map[String, Set[String]] = detectLanguage(text)

    // TODO delete me
    println(s"==> detected languages: $detectedLanguage")

    // must be a single element list.
    if(detectedLanguage.size != 1)
      Map(English-> Set())
    else {
      detectedLanguage
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

  def getKeyFromDictionaryValue(toBeReplaced: Array[String]) = {

    val dictionary: Map[String, Any] = loadDictionary(English)

    // contains all the retrieved keys
    val indexedDictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet.zipWithIndex

    val searchArray: Set[Int] = indexedDictionaryValues.map(e => if(e._1.contains(toBeReplaced)) e._2 else -1)

    searchArray.filter(_ >= 0)

    // contains all the retrieved values
    val dictionaryValues = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet

  }

  def translateBySentence(text: String, sourceLanguageInfo: Map[String, Set[String]]) = {
    val language = sourceLanguageInfo.keySet.head
    // FIXME already there
    //val matches = searchForLanguageMatches(text, language, useTokens = false)

    println(s"Translate sentence on matches: $sourceLanguageInfo")

    val strPattern = sourceLanguageInfo.values.flatten.head.r
    //    val pattern = """\d+""".r
    val matchingGroup = strPattern.findAllIn(text).toList.head

    val groupTokens = matchingGroup.split(" ")
    val toBeReplaced = groupTokens.takeRight(groupTokens.size - 1)

    val key = getKeyFromDictionaryValue(toBeReplaced)


    //    val strPattern = sourceLanguageInfo.values.flatten.head
    //    val pattern: Regex = new Regex(strPattern)
    //
    //    val group = pattern.findAllIn(text).matchData

    //    foreach {
    //      m => m.group(1)
    //    }

    println(s"group: $matchingGroup")

    (true, "")
  }

  private def translateTokens(text: String, sourceLanguageInfo: Map[String, Set[String]]) = {

    if (!sourceLanguageInfo.keySet.head.equals(English)) {
      val sourceLanguageDictionary: Map[String, Any] = loadDictionary(sourceLanguageInfo.keySet.head)

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
   *
   * @param text the text to translate.
   * @param sourceLanguageInfo the source language.
   * @param destination the destination language.
   * @return the translated text from source language to destination language.
   * */
  private def _translate(text: String,
                         sourceLanguageInfo: Map[String, Set[String]],
                         destination: String = English) = {


    // TODO QUI per regex translation
    println("_translate...")
    println(s"SOURCE LANG: $sourceLanguageInfo")
    println(s"DESTINATION LANG: $destination")

    // Map(it -> Set((.*) anni fa))
    if(!sourceLanguageInfo.keySet.head.isEmpty
      && !sourceLanguageInfo.head._2.isEmpty && sourceLanguageInfo.head._2.toString().split(" ").size != 1){

      translateBySentence(text, sourceLanguageInfo)
      //      text
    }
    else {
      translateTokens(text, sourceLanguageInfo)
    }
    ""
  }

  /**
   * Translate the text from source language to destination language.
   *
   * @param text the text to translate.
   * @param source the source language.
   * @param destination the destination language.
   * @return the translated text from source language to destination language.
   * */
  def translate(text: String, source: String, destination: String = English): String = {

    // 1. set or detect source language
    val _sourceLanguageInfo: Map[String, Set[String]] = detectSourceLanguage(text)

    // 2. apply translation
    val translated = _translate(text, _sourceLanguageInfo, destination)

    translated
  }
}
