package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.spell.util.Utilities
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, WithAnnotate}
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashSet


/** This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary. Inspired by Norvig model
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala]] for further reference on how to use this API
  *
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class NorvigSweetingModel(override val uid: String) extends AnnotatorModel[NorvigSweetingModel] with WithAnnotate[NorvigSweetingModel] with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */

  def this() = this(Identifiable.randomUID("SPELL"))

  private val logger = LoggerFactory.getLogger("NorvigApproach")

  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = TOKEN
  /** Input annotator type : TOKEN
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** @group param */
  protected val wordCount: MapFeature[String, Long] = new MapFeature(this, "wordCount")

  /** @group getParam */
  protected def getWordCount: Map[String, Long] = $$(wordCount)

  /** @group setParam */
  def setWordCount(value: Map[String, Long]): this.type = set(wordCount, value)

  /** @group param */
  private lazy val allWords: HashSet[String] = {
    if ($(caseSensitive)) HashSet($$(wordCount).keys.toSeq:_*)
    else HashSet($$(wordCount).keys.toSeq.map(_.toLowerCase):_*)
  }

  /** @group param */
  private lazy val frequencyBoundaryValues: (Long, Long) = {
    val min: Long = $$(wordCount).filter(_._1.length > $(wordSizeIgnore)).minBy(_._2)._2
    val max = $$(wordCount).filter(_._1.length > $(wordSizeIgnore)).maxBy(_._2)._2
    (min, max)
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
      val verifiedWord = checkSpellWord(token.result)
      Annotation(
        outputAnnotatorType,
        token.begin,
        token.end,
        verifiedWord._1,
        Map(
          "confidence"->verifiedWord._2.toString,
          "sentence"->token.metadata("sentence")
        )
      )
    }
  }

  def checkSpellWord(raw: String): (String, Double) = {
    val input = Utilities.limitDuplicates($(dupsLimit), raw)
    logger.debug(s"spell checker target word: $input")
    val possibility = getBestSpellingSuggestion(input)
    if (possibility._1.isDefined) return (possibility._1.get, possibility._2)

    val listedSuggestions = suggestions(input)
    val sortedFrequencies = getSortedWordsByFrequency(listedSuggestions, input)
    val sortedHamming = getSortedWordsByHamming(listedSuggestions, input)
    getResult(sortedFrequencies, sortedHamming, input)
  }

  private def getBestSpellingSuggestion(word: String): (Option[String], Double) = {
    var score: Double = 0
    if ($(shortCircuit)) {
      val suggestedWord = getShortCircuitSuggestion(word).getOrElse(word)
      score = getScoreFrequency(suggestedWord)
      (Some(suggestedWord), score)
    } else {
      val suggestions = getSuggestion(word: String)
      (suggestions._1, suggestions._2)
    }
  }

  private def getShortCircuitSuggestion(word: String): Option[String] = {
    if (Utilities.reductions(word, $(reductLimit)).exists(allWords.contains)) Some(word)
    else if (Utilities.getVowelSwaps(word, $(vowelSwapLimit)).exists(allWords.contains)) Some(word)
    else if (Utilities.variants(word).exists(allWords.contains)) Some(word)
    else if (both(word).exists(allWords.contains)) Some(word)
    else if ($(doubleVariants) && computeDoubleVariants(word).exists(allWords.contains)) Some(word)
    else None
  }

  /** variants of variants of a word */
  def computeDoubleVariants(word: String): List[String] = Utilities.variants(word).flatMap(variant =>
    Utilities.variants(variant))

  private def getSuggestion(word: String): (Option[String], Double) = {
    if (allWords.contains(word)) {
      logger.debug("Word found in dictionary. No spell change")
      (Some(word), 1)
    } else if (word.length <= $(wordSizeIgnore)) {
      logger.debug("word ignored because length is less than wordSizeIgnore")
      (Some(word), 0)
    } else if (allWords.contains(word.distinct)) {
      logger.debug("Word as distinct found in dictionary")
      val score = getScoreFrequency(word.distinct)
      (Some(word.distinct), score)
    } else (None, -1)
  }

  def getScoreFrequency(word: String): Double = {
    val frequency = Utilities.getFrequency(word, $$(wordCount))
    normalizeFrequencyValue(frequency)
  }

  def normalizeFrequencyValue(value: Long): Double = {
    if (value > frequencyBoundaryValues._2) {
      return 1
    }
    if (value < frequencyBoundaryValues._1) {
      return 0
    }
    val normalizedValue = (value - frequencyBoundaryValues._1).toDouble / (frequencyBoundaryValues._2 - frequencyBoundaryValues._1).toDouble
    BigDecimal(normalizedValue).setScale(4, BigDecimal.RoundingMode.HALF_UP).toDouble
  }

  private def suggestions(word: String): List[String] = {
    val intersectedPossibilities = allWords.intersect({
      val base =
        Utilities.reductions(word, $(reductLimit)) ++
          Utilities.getVowelSwaps(word, $(vowelSwapLimit)) ++
          Utilities.variants(word) ++
          both(word)
      if ($(doubleVariants)) base ++ computeDoubleVariants(word) else base
    }.toSet)
    if (intersectedPossibilities.nonEmpty) intersectedPossibilities.toList
    else List.empty[String]
  }

  private def both(word: String): List[String] = {
    Utilities.reductions(word, $(reductLimit)).flatMap(reduction => Utilities.getVowelSwaps(reduction, $(vowelSwapLimit)))
  }

  def getSortedWordsByFrequency(words: List[String], input: String): List[(String, Long)] = {
    val filteredWords = words.withFilter(_.length >= input.length)
    val sortedWordsByFrequency = filteredWords.map(word => (word, compareFrequencies(word)))
      .sortWith(_._2 > _._2).take($(intersections))
    logger.debug(s"recommended by frequency: ${sortedWordsByFrequency.mkString(", ")}")
    sortedWordsByFrequency
  }

  private def compareFrequencies(value: String): Long = Utilities.getFrequency(value, $$(wordCount))

  def getSortedWordsByHamming(words: List[String], input: String): List[(String, Long)] = {
    val sortedWordByHamming = words.map(word => (word, compareHammers(input)(word)))
      .sortBy(_._2).takeRight($(intersections))
    logger.debug(s"recommended by hamming: ${sortedWordByHamming.mkString(", ")}")
    sortedWordByHamming
  }

  private def compareHammers(input: String)(value: String): Long = Utilities.computeHammingDistance(input, value)

  def getResult(wordsByFrequency: List[(String, Long)], wordsByHamming: List[(String, Long)], input: String):
  (String, Double) = {
    var recommendation: (Option[String], Double) = (None, 0)
    val intersectWords = wordsByFrequency.map(word => word._1).intersect(wordsByHamming.map(word => word._1))
    if (wordsByFrequency.isEmpty && wordsByHamming.isEmpty) {
      logger.debug("no intersection or frequent words found")
      recommendation = (Some(input), 0)
    } else if (wordsByFrequency.isEmpty || wordsByHamming.isEmpty) {
      logger.debug("no intersection but one recommendation found")
      recommendation = getRecommendation(wordsByFrequency, wordsByHamming)
    } else if (intersectWords.nonEmpty) {
      logger.debug("hammer and frequency recommendations found")
      val frequencyAndHammingRecommendation = getFrequencyAndHammingRecommendation(wordsByFrequency, wordsByHamming,
        intersectWords)
      recommendation = (frequencyAndHammingRecommendation._1, frequencyAndHammingRecommendation._2)
    } else {
      logger.debug("no intersection of hammer and frequency")
      recommendation = getFrequencyOrHammingRecommendation(wordsByFrequency, wordsByHamming, input)
    }
    (recommendation._1.getOrElse(input), recommendation._2)
  }

  private def getRecommendation(wordsByFrequency: List[(String, Long)], wordsByHamming: List[(String, Long)]) = {
    if (wordsByFrequency.nonEmpty) {
      getResultByFrequency(wordsByFrequency)
    } else {
      getResultByHamming(wordsByHamming)
    }
  }

  private def getFrequencyAndHammingRecommendation(wordsByFrequency: List[(String, Long)],
                                                   wordsByHamming: List[(String, Long)],
                                                   intersectWords: List[String]): (Option[String], Double) = {
    val wordsByFrequencyAndHamming = intersectWords.map{word =>
      val frequency = wordsByFrequency.find(_._1 == word).get._2
      val hamming = wordsByHamming.find(_._1 == word).get._2
      (word, frequency, hamming)
    }
    val bestFrequencyValue = wordsByFrequencyAndHamming.maxBy(_._2)._2
    val bestHammingValue = wordsByFrequencyAndHamming.minBy(_._3)._3
    val bestRecommendations = wordsByFrequencyAndHamming.filter(word =>
      word._2 == bestFrequencyValue && word._3 == bestHammingValue)
    if (bestRecommendations.nonEmpty) {
      val result = (Utilities.getRandomValueFromList(bestRecommendations),
        Utilities.computeConfidenceValue(bestRecommendations))
      (Some(result._1.get._1), result._2)
    } else {
      if ($(frequencyPriority)) {
        (Some(wordsByFrequencyAndHamming.sortBy(_._3).maxBy(_._2)._1), 1.toDouble)
      } else {
        (Some(wordsByFrequencyAndHamming.sortBy(_._2).reverse.minBy(_._3)._1), 1.toDouble)
      }
    }
  }

  def getResultByFrequency(wordsByFrequency: List[(String, Long)]): (Option[String], Double) = {
    val bestFrequencyValue = wordsByFrequency.maxBy(_._2)._2
    val bestRecommendations = wordsByFrequency.filter(_._2 == bestFrequencyValue).map(_._1)
    (Utilities.getRandomValueFromList(bestRecommendations), Utilities.computeConfidenceValue(bestRecommendations))
  }

  def getResultByHamming(wordsByHamming: List[(String, Long)]): (Option[String], Double) = {
    val bestHammingValue = wordsByHamming.minBy(_._2)._2
    val bestRecommendations = wordsByHamming.filter(_._2 == bestHammingValue).map(_._1)
    (Utilities.getRandomValueFromList(bestRecommendations), Utilities.computeConfidenceValue(bestRecommendations))
  }

  def getFrequencyOrHammingRecommendation(wordsByFrequency: List[(String, Long)], wordsByHamming: List[(String, Long)],
                                          input: String): (Option[String], Double) = {
    val frequencyResult: String = getResultByFrequency(wordsByFrequency)._1.getOrElse(input)
    val hammingResult: String = getResultByHamming(wordsByHamming)._1.getOrElse(input)
    var result =  List(frequencyResult, hammingResult)
    if (frequencyResult == input) {
      result = List (hammingResult)
    } else if (hammingResult == input) {
      result = List (frequencyResult)
    }

    (Utilities.getRandomValueFromList(result), Utilities.computeConfidenceValue(result))
  }

}

trait ReadablePretrainedNorvig extends ParamsAndFeaturesReadable[NorvigSweetingModel] with HasPretrained[NorvigSweetingModel] {
  override val defaultModelName = Some("spellcheck_norvig")
  /** Java compliant-overrides */
  override def pretrained(): NorvigSweetingModel = super.pretrained()
  override def pretrained(name: String): NorvigSweetingModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): NorvigSweetingModel = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): NorvigSweetingModel = super.pretrained(name, lang, remoteLoc)
}

object NorvigSweetingModel extends ReadablePretrainedNorvig