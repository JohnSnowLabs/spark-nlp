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

  val EmptyStr = ""
  val KeyPlaceholder = "#K#"
  val ValuePlaceholder = "#V#"
  val DigitsPattern = """(\\d+)"""
  val NotDetected = "-1"

  /**
    * Load dictionary from supported language repository.
    * @param language the language dictionary to load. Default is English.
    * @return a map containing the language dictionary or throws an exception.
    * */
  def loadDictionary(language: String = English) = {
    val DictionaryPath = s"$TranslationDataBaseDir$language$JsonSuffix"

    var jsonString = EmptyStr;
    try{
      jsonString = Source.fromFile(DictionaryPath).mkString
    } catch {
      case e: FileNotFoundException => throw new Exception(s"Couldn't find $language file in repository.")
      case e: IOException => throw new Exception("Got an IOException!")
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

  /**
    * Load available language keys to process further matches
    * @param text: text to match language against.
    * @return a map containing the matching languages.
    * */
  def _processSourceLanguageInfo(text: String, sourceLanguage: String) = {
    val supportedLanguages =
      scala.io.Source
        .fromFile(SupportedLanguagesFilePath, Encoding)
        .getLines.toList

    if(!sourceLanguage.isEmpty) {
      val actualLanguage = List(sourceLanguage)

      val matchingLanguages = actualLanguage
        .map(l => searchForLanguageMatches(text, l))
        .filterNot(_._1.equals(NotFound))
        .toMap

      matchingLanguages
    }else {
      // TODO Autodetection flow or Exception
      val activeLanguages = supportedLanguages.filterNot(_.startsWith(SkipChar)) // skip char

      val matchingLanguages = activeLanguages
        .map(l => searchForLanguageMatches(text, l))
        .filterNot(_._1.equals(NotFound))
        .toMap

      matchingLanguages
    }
  }

  def stringValuesToRegexes(regexValuesIntersection: Set[String]): Set[Regex] = {
    val res = regexValuesIntersection
      .map(_.replaceAll(ValuePlaceholder, DigitsPattern))
      .map(_.toLowerCase)
      .map(s => s.r)
    res
  }

  /**
    * Search for language matches token by token.
    *
    * @param text: the text to process for matching.
    * @param language: the 2 characters string identifying a supported language.
    * @return a tuple representing language matches information, i.e. (language, Set(matches))
    * */
  private def searchForLanguageMatches(text: String, language: String) = {
    val dictionary: Map[String, Any] = loadDictionary(language)

    // contains all the retrieved values
    val candidates = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet

    val tokenMatches = findTokenMatches(text, candidates)
    val sentenceMatches = findSentenceMatches(text, candidates)
    val matches = tokenMatches.union(sentenceMatches.map(_._2.toString))

    // skipping single character matches
    val filteredMatches = matches.filter(_.length > 1)

    val matchingLanguages: (String, Set[String]) =
      if(filteredMatches.size != 0) {
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse(NameKey, NotAvailable)
          .toString -> filteredMatches
      } else
        NotFound -> Set.empty

    matchingLanguages
  }

  private def findSentenceMatches(text: String, dictionaryValues: Set[String]) = {

    val regexes = stringValuesToRegexes(dictionaryValues)

    val sentenceMatches =
      for (regex <- regexes;
           res = regex.findFirstMatchIn(text.toLowerCase)
           if !res.isEmpty && res.size != 0
           ) yield (text, regex)

    sentenceMatches
  }

  private def findTokenMatches(text: String, dictionaryValues: Set[String]) = {
    val candidateTokens =
      text
        .split(SpaceChar)
        .map(_.toLowerCase)
        .toSet

    val tokenMatches =
      candidateTokens
        .intersect(
          dictionaryValues.filterNot(_.contains(ValuePlaceholder)))

    tokenMatches
  }

  /**
    * Process date matches with the source language.
    *
    * @param text: the text to detect.
    * @return the detected language as tuple of matched languages with their frequency.
    * */
  def processSourceLanguageInfo(text: String, sourceLanguage: String): Map[String, Set[String]] = {
    // e.g. Map(it -> Set((.*) anni fa))
    val processedSourceLanguageInfo: Map[String, Set[String]] = _processSourceLanguageInfo(text, sourceLanguage)

    // if multiple source languages match we default to english as further processing is not possible
    if(processedSourceLanguageInfo.size != 1)
      Map(English-> Set())
    else {
      processedSourceLanguageInfo
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

      val getListifiedValues: String => List[String] = dictionary.getOrElse(_, NotAvailable) match {
        case l: List[String @unchecked] => l
        case s: String => List(s)
        case m: Map[String @unchecked, Any @unchecked] => m.keySet.toList
        case _ => throw new Exception("Cannot listify dictionary value.")
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

  def searchKeyFromValuesMatch(dictionary: Map[String, Any], k: String, toBeReplaced: String) = {

    val candidates: List[String] = dictionary.getOrElse(k, List(NotFound)) match {
      case i: List[String @unchecked] => i
      case _ => List(dictionary.getOrElse(k, List(NotFound)).toString)
    }

    val cardinality: Int = getTimeUnitCardinality(toBeReplaced)

    val res =
      for(c <- candidates
          if c.contains(toBeReplaced) || c.replaceAll(ValuePlaceholder, cardinality.toString).contains(toBeReplaced)
          ) yield c

    if(!res.isEmpty)
      (k, res.head)
    else
      (k, NotFound)
  }

  private def getTimeUnitCardinality(text: String) = {
    val pattern = """(\d+)""".r
    val cardinality = pattern.findFirstIn(text) match {
      case Some(m) => m.toInt
      case _ => -1
    }
    cardinality
  }

  def getKeyFromDictionaryValue(toBeReplaced: Array[String], sourceLanguage: String) = {

    val dictionary: Map[String, Any] = loadDictionary(sourceLanguage)

    // contains all the retrieved keys
    val iterKeys: Iterable[String] = dictionary
      .asInstanceOf[Map[String, Any]].keys

    val replacingKey: Iterable[(String, String)] = iterKeys
      .map(k => searchKeyFromValuesMatch(dictionary, k, toBeReplaced.mkString(SpaceChar)))
      .filterNot(_._2.equals(NotFound))

    replacingKey
  }

  def adjustPlurality(text: String) = {
    val pattern = """(\d+)""".r
    val res = pattern.findFirstIn(text) match {
      case Some(m) => m.toInt
      case _ => -1
    }

    val candidates = Array("second", "minute", "hour", "day", "month", "year")

    var acc = text
    val adjusted: String =
      if(res > 1) {
        val matchingCandidates = for(c <- candidates if text.contains(c)) yield c
        matchingCandidates.foreach(c => acc = acc.replaceAll(c, c + "s"))
        acc
      }
      else {
        text
      }

    adjusted
  }

  /**
    * Translate sentence from source info.
    * @param text: sentence to translate.
    * @param sourceLanguageInfo: the source info map.
    * @return the translated sentence.
    * */
  def translateBySentence(text: String, sourceLanguageInfo: Map[String, Set[String]]) = {

    val strPattern = sourceLanguageInfo.values.flatten.head.r
    val normalizedText = text.toLowerCase
    val matchingGroup = strPattern.findAllIn(normalizedText).toList.head

    val toBeReplaced =
      if(!matchingGroup.contains(ValuePlaceholder) || !matchingGroup.contains(ValuePlaceholder))
        Array(matchingGroup)
      else {
        val groupTokens = matchingGroup.split(SpaceChar)
        groupTokens.takeRight(groupTokens.size - 1)
      }

    val sourceLanguage = sourceLanguageInfo.head._1
    val replacingKeys = getKeyFromDictionaryValue(toBeReplaced, sourceLanguage)

    val cardinality = getTimeUnitCardinality(text).toString

    var acc = EmptyStr

    if(cardinality != NotDetected){
      replacingKeys.foreach(rk =>
        acc = text.replaceAll(
          rk._2.replace(ValuePlaceholder, cardinality),
          rk._1.replace(KeyPlaceholder, cardinality)))

      val adjusted = adjustPlurality(acc)
      adjusted
    }
    else {
      replacingKeys
        .foreach(rk =>
          acc = normalizedText.replaceAll(rk._2, rk._1))
      acc
    }
  }

  /**
    * Translate tokens from source info token by token.
    * @param text: sentence to translate.
    * @param sourceLanguageInfo: the osurce info map.
    * @return the translated sentence.
    * */
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

    val key = sourceLanguageInfo.keySet.head
    val longestMatch = sourceLanguageInfo.values.flatten.toList
      .sortWith(_.length > _.length)
      .head
    val _sourceLanguageInfo: Map[String, Set[String]] = Map(key -> Set(longestMatch))

    val predicates = Array(
      !_sourceLanguageInfo.keySet.head.isEmpty,
      !_sourceLanguageInfo.head._2.isEmpty,
      _sourceLanguageInfo.head._2.toString().split(SpaceChar).size != 1
    )

    val res =
      if(predicates.forall(_.equals(true)))
        translateBySentence(text, _sourceLanguageInfo)
      else
        translateTokens(text, _sourceLanguageInfo)

    res
  }

  /**
    * Translate the text from source language to destination language.
    *
    * @param text the text to translate.
    * @param sourceLanguage the source language.
    * @param destination the destination language.
    * @return the translated text from source language to destination language.
    * */
  def translate(text: String, sourceLanguage: String, destination: String = English): String = {
    // 1. detect source language
    val _sourceLanguageInfo: Map[String, Set[String]] = processSourceLanguageInfo(text, sourceLanguage)

    // 2. apply translation if source is not english
    val translated =
      if(!_sourceLanguageInfo.keySet.head.equals(English))
        _translate(text, _sourceLanguageInfo, destination)
      else
        text

    translated
  }
}