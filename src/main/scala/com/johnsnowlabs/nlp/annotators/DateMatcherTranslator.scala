/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.util.JsonParser
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.{FileNotFoundException, IOException}
import scala.io.Source
import scala.util.matching.Regex

sealed trait DateMatcherTranslatorPolicy {
  def value: String
}


case object SingleDatePolicy extends DateMatcherTranslatorPolicy {
  override def value: String = "single"
}


case object MultiDatePolicy extends DateMatcherTranslatorPolicy {
  override def value: String = "multi"
}


class DateMatcherTranslator(policy: DateMatcherTranslatorPolicy) extends Serializable {

  val SupportedLanguagesFilePath = "/date-matcher/supported_languages.txt"
  val TranslationDataBaseDir = "/date-matcher/translation-dictionaries/dynamic/"

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

  val AccentsPattern = "\\p{InCombiningDiacriticalMarks}+"
  val PunctuationPattern = "[^a-zA-Z0-9 \\/]"

  /** Return one policy in [single, multi] associated to this translator */
  def getPolicy() = policy

  /**
   * Load dictionary from supported language repository.
   *
   * @param language the language dictionary to load. Default is English.
   * @return a map containing the language dictionary or throws an exception.
   * */
  def loadDictionary(language: String = English) = {
    val DictionaryPath = s"$TranslationDataBaseDir$language$JsonSuffix"

    var jsonString = EmptyStr;
    try {
      jsonString = Source.fromInputStream(getClass.getResourceAsStream(DictionaryPath)).mkString
    } catch {
      case e: FileNotFoundException => throw new Exception(s"Couldn't find $language file in repository.")
      case e: IOException => throw new Exception("Got an IOException!")
    }

    val jsonMap = JsonParser.parseObject[Map[String, Any]](jsonString)
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
   *
   * @param text : text to match language against.
   * @return a map containing the matching languages.
   * */
  def _processSourceLanguageInfo(text: String, sourceLanguage: String) = {
    val supportedLanguages =
      Source.fromInputStream(getClass.getResourceAsStream(SupportedLanguagesFilePath))
        .getLines()
        .toList

    if (!sourceLanguage.isEmpty) {
      val actualLanguage = List(sourceLanguage)

      val matchingLanguages = actualLanguage
        .map(l => searchForLanguageMatches(text, l))
        .filterNot(_._1.equals(NotFound))
        .toMap

      matchingLanguages
    } else {
      // TODO Autodetection flow or Exception
      val activeLanguages = supportedLanguages.filterNot(_.startsWith(SkipChar)) // skip char

      val matchingLanguages = activeLanguages
        .map(l => searchForLanguageMatches(text, l))
        .filterNot(_._1.equals(NotFound))
        .toMap

      matchingLanguages
    }
  }

  def stringValuesToRegexes(regexValuesIntersection: Set[String], sourceLanguage: String): Set[Regex] = {
    val res = regexValuesIntersection
      .map(_.replaceAll(ValuePlaceholder, DigitsPattern))
      .filterNot(_.equals(sourceLanguage))
      .map(_.toLowerCase)
      .map(s => s.r)
    res
  }

  /**
   * Search for language matches token by token.
   *
   * @param text           : the text to process for matching.
   * @param sourceLanguage : the 2 characters string identifying a supported language.
   * @return a tuple representing language matches information, i.e. (language, Set(matches))
   * */
  private def searchForLanguageMatches(text: String, sourceLanguage: String) = {
    val dictionary: Map[String, Any] = loadDictionary(sourceLanguage)

    // contains all the retrieved values
    val candidates = dictionary
      .asInstanceOf[Map[String, Any]]
      .values.flatten(listify).asInstanceOf[List[String]]
      .toSet

    val tokenMatches = findTokenMatches(text, candidates)
    val sentenceMatches = findSentenceMatches(text, candidates, sourceLanguage)
    val matches = tokenMatches.union(sentenceMatches.map(_._2.toString))

    // skipping single character matches
    val filteredMatches = matches.filter(_.length > 1)

    val matchingLanguages: (String, Set[String]) =
      if (filteredMatches.size != 0) {
        dictionary
          .asInstanceOf[Map[String, Any]]
          .getOrElse(NameKey, NotAvailable)
          .toString -> filteredMatches
      } else
        NotFound -> Set.empty

    matchingLanguages
  }

  private def findSentenceMatches(text: String, dictionaryValues: Set[String], sourceLanguage: String) = {

    val regexes = stringValuesToRegexes(dictionaryValues, sourceLanguage)

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
   * @param text : the text to detect.
   * @return the detected language as tuple of matched languages with their frequency.
   * */
  def processSourceLanguageInfo(text: String, sourceLanguage: String): Map[String, Set[String]] = {
    // e.g. Map(it -> Set((.*) anni fa))
    val processedSourceLanguageInfo: Map[String, Set[String]] = _processSourceLanguageInfo(text, sourceLanguage)

    // if multiple source languages match we default to english as further processing is not possible
    if (processedSourceLanguageInfo.size != 1) {
      // nothing to translate has matched
      Map(English -> Set())
    } else {
      processedSourceLanguageInfo
    }
  }

  // utility type
  type DateMatcherIndexedToken = (String, Int)

  private def adjustPunctuation(text: String): String = {
    text.replaceAll("\\.|,|!|\\?|:", EmptyStr)
  }


  /**
   * Matches the indexed text token against the passed dictionary.
   *
   * @param indexToken the indexed token to match.
   * @param dictionary the dictionary to match token against.
   * */
  def matchIndexedToken: (DateMatcherIndexedToken, Map[String, Any]) => DateMatcherIndexedToken =
    (indexToken, dictionary) => {
      val (token, index) = indexToken
      val keys = dictionary.keySet

      val getListifiedValues: String => List[String] = dictionary.getOrElse(_, NotAvailable) match {
        case l: List[String@unchecked] => l
        case s: String => List(s)
        case m: Map[String@unchecked, Any@unchecked] => m.keySet.toList
        case _ => throw new Exception("Cannot listify dictionary value.")
      }

      val translated: Set[(String, Int)] =
        for (k <- keys
             if getListifiedValues(k).contains(token)
             ) yield (k, index)

      // only first match if any is returned
      if (!translated.isEmpty)
        translated.head
      else {
        // bypassed as not matched
        indexToken
      }
    }

  /**
   * Apply translation switching token where an index has been translated using the dictionary matching.
   *
   * @param translatedIndexedToken : the translated index token to replace in text.
   * @param text                   : the original text where translation is applied.
   * @return the text translated using token replacement.
   * */
  def applyTranslation(translatedIndexedToken: Array[(String, Int)], text: String) = {
    val tokens = text.split(SpaceChar)
    translatedIndexedToken.map(t => tokens(t._2) = t._1)
    tokens.mkString(" ")
  }

  def searchKeyFromValuesMatch(dictionary: Map[String, Any], k: String, toBeReplaced: String) = {

    val candidates: List[String] = dictionary.getOrElse(k, List(NotFound)) match {
      case i: List[String@unchecked] => i
      case _ => List(dictionary.getOrElse(k, List(NotFound)).toString)
    }

    val cardinality: Int = getTimeUnitCardinality(toBeReplaced)

    val res =
      for (c <- candidates
           if c.contains(toBeReplaced) || c.replaceAll(ValuePlaceholder, cardinality.toString).contains(toBeReplaced)
           ) yield c

    if (!res.isEmpty)
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

    val candidates = Array("second", "minute", "hour", "day", "week", "month", "year")

    var acc = text.replaceAll("\\.|,", " ")
    val adjusted: String =
      if (res > 1) {
        val matchingCandidates = for (c <- candidates
                                      if text.contains(c)
                                        && !text.contains(1 + " " + c)
                                        && !text.contains(c + "s")) yield c
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
   *
   * @param text               : sentence to translate.
   * @param sourceLanguageInfo : the source info map.
   * @return the translated sentence.
   * */
  def translateBySentence(text: String,
                          sourceLanguageInfo: Map[String, Set[String]],
                          policy: DateMatcherTranslatorPolicy) = {

    val _text = text.toLowerCase

    // return only the first matched pattern
    def translateWithMultiDatePolicy = {
      val patterns: Iterable[Regex] = sourceLanguageInfo.values.flatten.map(_.r)
      var acc = text

      for (p <- patterns) {
        acc = _translateWithPattern(acc, p)
      }

      acc
    }

    def _translateWithPattern(text: String, pattern: Regex) = {
      val matchingGroup = pattern.findAllIn(text).toList.head

      val toBeReplaced =
        if (!matchingGroup.contains(ValuePlaceholder) || !matchingGroup.contains(ValuePlaceholder))
          Array(matchingGroup)
        else {
          val groupTokens = matchingGroup.split(SpaceChar)
          groupTokens.takeRight(groupTokens.size - 1)
        }

      val sourceLanguage = sourceLanguageInfo.head._1
      val replacingKeys = getKeyFromDictionaryValue(toBeReplaced, sourceLanguage)

      val cardinality = getTimeUnitCardinality(text).toString

      var acc = EmptyStr

      if (cardinality != NotDetected) {
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
            acc = _text.replaceAll(rk._2, rk._1))
        acc
      }
    }

    // return only the first matched pattern
    def translateWithSingleDatePolicy = {
      val strPattern = sourceLanguageInfo.values.flatten.head.r
      _translateWithPattern(_text, strPattern)
    }

    val res = policy.value match {
      case "single" => translateWithSingleDatePolicy
      case "multi" => translateWithMultiDatePolicy
      case _ => throw new Exception("Unknown date match translation policy")
    }

    res
  }

  /**
   * Translate tokens from source info token by token.
   *
   * @param text               : sentence to translate.
   * @param sourceLanguageInfo : the osurce info map.
   * @return the translated sentence.
   * */
  private def translateTokens(text: String, sourceLanguageInfo: Map[String, Set[String]]) = {

    // tokens can have punctuation so we remove it
    val _text = adjustPunctuation(text)

    if (!sourceLanguageInfo.keySet.head.equals(English)) {
      val sourceLanguageDictionary: Map[String, Any] = loadDictionary(sourceLanguageInfo.keySet.head)

      val translatedIndexedToken: Array[DateMatcherIndexedToken] =
        _text
          .split(SpaceChar).zipWithIndex
          .map(matchIndexedToken(_, sourceLanguageDictionary))

      applyTranslation(translatedIndexedToken, _text)
    }
    else
      _text
  }

  /**
   * Return the longest match
   *
   * @param sortedMatches : the list of matches sorted by length
   * @return a List containing the longest match found in the input list
   * */
  private def _regularizeSingleDateMatches(sortedMatches: List[String]) = {
    List(sortedMatches.head)
  }

  /**
   * Return the longest match
   *
   * @param sortedMatches : the list of matches sorted by length
   * @param n             : the number of desire matches
   * @return the n longest match found in the input list
   * */
  private def _regularizeMultiDateMatches(sortedMatches: List[String], n: Int = 2) = {
    var acc = sortedMatches

    for (m <- sortedMatches) {
      sortedMatches.map(key =>
        if (m.contains(key) && m.length > key.length)
          acc = acc.filterNot(_ == key))
    }

    acc
  }

  def preprocessSortedByLengthMatches(sortedMatches: List[String], policy: DateMatcherTranslatorPolicy) = {
    policy.value match {
      case "single" => _regularizeSingleDateMatches(sortedMatches)
      case "multi" => _regularizeMultiDateMatches(sortedMatches)
    }
  }

  /**
   * Translate the text from source language to destination language.
   *
   * @param text               the text to translate.
   * @param sourceLanguageInfo the source language.
   * @param destination        the destination language.
   * @return the translated text from source language to destination language.
   * */
  private def _translate(text: String,
                         sourceLanguageInfo: Map[String, Set[String]],
                         destination: String = English) = {

    val key = sourceLanguageInfo.keySet.head

    val sortedMatches: List[String] = sourceLanguageInfo.values.flatten.toList.sortWith(_.length > _.length)

    // Date Matcher takes 1 match, multi date n = 2 matches (default)
    val _sortedMatches = preprocessSortedByLengthMatches(sortedMatches, getPolicy)

    val _sourceLanguageInfo: Map[String, Set[String]] = Map(key -> _sortedMatches.toSet)

    val predicates = Array(
      !_sourceLanguageInfo.keySet.head.isEmpty,
      !_sourceLanguageInfo.head._2.isEmpty,
      _sourceLanguageInfo.head._2.toString().split(SpaceChar).size != 1
    )

    val res =
      if (predicates.forall(_.equals(true)))
        translateBySentence(text, _sourceLanguageInfo, getPolicy)
      else
        translateTokens(text, _sourceLanguageInfo)

    res
  }

  /**
   * Translate the text from source language to destination language.
   *
   * @param _text          the text to translate.
   * @param sourceLanguage the source language.
   * @param destination    the destination language.
   * @return the translated text from source language to destination language.
   * */
  def translate(text: String, sourceLanguage: String, destination: String = English): String = {

    // 0. normalize
    val _text = text.toLowerCase

    // 1. detect source language
    val _sourceLanguageInfo: Map[String, Set[String]] = processSourceLanguageInfo(_text, sourceLanguage)

    // 2. apply translation if source is not english
    val translated =
      if (!_sourceLanguageInfo.keySet.head.equals(English))
        _translate(_text, _sourceLanguageInfo, destination)
      else
        _text

    translated
  }
}