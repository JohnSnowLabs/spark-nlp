package com.johnsnowlabs.nlp.annotators

import org.json4s._
import org.json4s.jackson.JsonMethods._
import sun.font.TextSourceLabel

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

  /**
   * Load available language keys to process further matches
   * @param text: text to match language against.
   * @return a map containing the matching languages.
   * */
  def detectSourceLanguageInfo(text: String) = {
    val supportedLanguages =
      scala.io.Source
        .fromFile(SupportedLanguagesFilePath, Encoding)
        .getLines.toList

    val activeLanguages = supportedLanguages.filterNot(_.startsWith(SkipChar)) // skip char

    val matchingLanguages = activeLanguages
      .map(l => searchForLanguageMatches(text, l))
      .filterNot(_._1.equals(NotFound))
      .toMap

    matchingLanguages
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

    val matchingLanguages: (String, Set[String]) =
      if(matches.size != 0) {
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse(NameKey, NotAvailable)
          .toString -> matches
      } else
        NotFound -> Set.empty

    matchingLanguages
  }

  private def findSentenceMatches(text: String, dictionaryValues: Set[String]) = {
    val regexValuesIntersection = dictionaryValues.filter(_.contains("#V#"))

    val regexes = stringValuesToRegexes(regexValuesIntersection)

    val sentenceMatches =
      for (regex <- regexes;
           res = regex.findFirstMatchIn(text)
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
          dictionaryValues.filterNot(_.contains("#V#")))

    tokenMatches
  }

  /**
   * Detect the source language by looking for date language matches.
   *
   * @param text: the text to detect.
   * @return the detected language as tuple of matched languages with their frequency.
   * */
  def detectSourceLanguage(text: String): Map[String, Set[String]] = {
    // e.g. Map(it -> Set((.*) anni fa))
    val detectedSourceLanguageInfo: Map[String, Set[String]] = detectSourceLanguageInfo(text)

    // if multiple source languages match we default to english as further processing is not possible
    if(detectedSourceLanguageInfo.size != 1)
      Map(English-> Set())
    else {
      detectedSourceLanguageInfo
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

  def searchKeyFromValuesMatch(dictionary: Map[String, Any], k: String, toBeReplaced: String) = {

    val candidates: List[String] = dictionary.getOrElse(k, List("NF")) match {
      case i: List[String] => i
      case _ => List(dictionary.getOrElse(k, List("NF")).toString)
    }
    val res =
      for(c <- candidates
          if c.contains(toBeReplaced)
          ) yield c

    if(!res.isEmpty)
      (k, res.head)
    else
      (k, "NF")
  }

  def getKeyFromDictionaryValue(toBeReplaced: Array[String], sourceLanguage: String) = {

    val dictionary: Map[String, Any] = loadDictionary(sourceLanguage)

    // contains all the retrieved keys
    val iterKeys: Iterable[String] = dictionary
      .asInstanceOf[Map[String, Any]].keys

    val replacingKey: Iterable[(String, String)] = iterKeys
      .map(k => searchKeyFromValuesMatch(dictionary, k, toBeReplaced.mkString(SpaceChar)))
      .filterNot(_._2.equals("NF"))

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
   * @param sourceLanguageInfo: the osurce info map.
   * @return the translated sentence.
   * */
  def translateBySentence(text: String, sourceLanguageInfo: Map[String, Set[String]]) = {

    val strPattern = sourceLanguageInfo.values.flatten.head.r
    val matchingGroup = strPattern.findAllIn(text).toList.head

    val groupTokens = matchingGroup.split(" ")
    val toBeReplaced = groupTokens.takeRight(groupTokens.size - 1)

    val sourceLanguage = sourceLanguageInfo.head._1
    val replacingKeys = getKeyFromDictionaryValue(toBeReplaced, sourceLanguage)

    var acc = ""
    replacingKeys.foreach(rk =>
      acc = text.replaceAll(
        rk._2.replace("#V#", ""),
        rk._1.replace("#K#", "")))

    adjustPlurality(acc)
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

    val predicates = Array(
      !sourceLanguageInfo.keySet.head.isEmpty,
      !sourceLanguageInfo.head._2.isEmpty,
      sourceLanguageInfo.head._2.toString().split(" ").size != 1
    )

    val res =
      if(predicates.forall(_.equals(true)))
        translateBySentence(text, sourceLanguageInfo)
      else
        translateTokens(text, sourceLanguageInfo)

    res
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
