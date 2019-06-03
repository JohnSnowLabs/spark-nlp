package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashSet

class NorvigSweetingModel(override val uid: String) extends AnnotatorModel[NorvigSweetingModel] with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */

  def this() = this(Identifiable.randomUID("SPELL"))
  private val logger = LoggerFactory.getLogger("NorvigApproach")

  override val outputAnnotatorType: AnnotatorType = TOKEN
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  protected val wordCount: MapFeature[String, Long] = new MapFeature(this, "wordCount")
  /** params */
  protected val wordSizeIgnore = new IntParam(this, "wordSizeIgnore", "minimum size of word before ignoring. Defaults to 3")
  protected val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  protected val reductLimit = new IntParam(this, "reductLimit", "word reductions limit. Defaults to 3")
  protected val intersections = new IntParam(this, "intersections", "hamming intersections to attempt. Defaults to 10")
  protected val vowelSwapLimit = new IntParam(this, "vowelSwapLimit", "vowel swap attempts. Defaults to 6")

  protected def getWordCount: Map[String, Long] = $$(wordCount)

  def setWordSizeIgnore(value: Int): this.type = set(wordSizeIgnore, value)
  def setDupsLimit(value: Int): this.type = set(dupsLimit, value)
  def setReductLimit(value: Int): this.type = set(reductLimit, value)
  def setIntersections(value: Int): this.type = set(intersections, value)
  def setVowelSwapLimit(value: Int): this.type = set(vowelSwapLimit, value)
  def setWordCount(value: Map[String, Long]): this.type = set(wordCount, value)

  private lazy val allWords: HashSet[String] = {
    if ($(caseSensitive)) HashSet($$(wordCount).keys.toSeq:_*)
    else HashSet($$(wordCount).keys.toSeq.map(_.toLowerCase):_*)
  }

  private lazy val frequencyBoundaryValues: (Long, Long) = {
    val min: Long = $$(wordCount).filter(_._1.length > $(wordSizeIgnore)).minBy(_._2)._2
    val max = $$(wordCount).filter(_._1.length > $(wordSizeIgnore)).maxBy(_._2)._2
    (min, max)
  }

  private def compareFrequencies(value: String): Long = Utilities.getFrequency(value, $$(wordCount))
  private def compareHammers(input: String)(value: String): Long = Utilities.computeHammingDistance(input, value)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
        val verifiedWord = checkSpellWord(token.result)
        Annotation(
          outputAnnotatorType,
          token.begin,
          token.end,
          verifiedWord._1,
          Map("score"->verifiedWord._2.toString)
        )
    }
  }

  def checkSpellWord(raw: String): (String, Double) = {
    val input = Utilities.limitDups($(dupsLimit), raw)
    logger.debug(s"spell checker target word: $input")
    val possibility = getBestSpellingSuggestion(input)
    if (possibility._1.isDefined) return (possibility._1.get, possibility._2)

    val listedSuggestions = suggestions(input)
    val sortedFrequencies = getSortedWordsByFrequency(listedSuggestions, input)
    val sortedHamming = getSortedWordsByHamming(listedSuggestions, input)
    (getResult(sortedFrequencies, sortedHamming, input), 0)
  }

  private def getBestSpellingSuggestion(word: String): (Option[String], Double) = {
    var suggestedWord: Option[String] = None
    if ($(shortCircuit)) {
      suggestedWord = getShortCircuitSuggestion(word)
    } else {
      suggestedWord = getSuggestion(word: String)
    }
    val score = getScoreFrequency(suggestedWord.getOrElse(""))
    (suggestedWord, score)
  }

  private def getShortCircuitSuggestion(word: String): Option[String] = {
    if (allWords.intersect(Utilities.reductions(word, $(reductLimit))).nonEmpty) Some(word)
    else if (allWords.intersect(Utilities.vowelSwaps(word, $(vowelSwapLimit))).nonEmpty) Some(word)
    else if (allWords.intersect(Utilities.variants(word)).nonEmpty) Some(word)
    else if (allWords.intersect(both(word)).nonEmpty) Some(word)
    else if ($(doubleVariants) && allWords.intersect(computeDoubleVariants(word)).nonEmpty) Some(word)
    else None
  }

  /** variants of variants of a word */
  def computeDoubleVariants(word: String): Set[String] = Utilities.variants(word).flatMap(variant =>
    Utilities.variants(variant))

  private def getSuggestion(word: String): Option[String] = {
    if (allWords.contains(word)) {
      logger.debug("Word found in dictionary. No spell change")
      Some(word)
    } else if (word.length <= $(wordSizeIgnore)) {
      logger.debug("word ignored because length is less than wordSizeIgnore")
      Some(word)
    } else if (allWords.contains(word.distinct)) {
      logger.debug("Word as distinct found in dictionary")
      Some(word.distinct)
    } else None
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
          Utilities.vowelSwaps(word, $(vowelSwapLimit)) ++
          Utilities.variants(word) ++
          both(word)
      if ($(doubleVariants)) base ++ computeDoubleVariants(word) else base
    })
    if (intersectedPossibilities.nonEmpty) intersectedPossibilities.toList
    else List.empty[String]
  }

  private def both(word: String): Set[String] = {
    Utilities.reductions(word, $(reductLimit)).flatMap(reduction => Utilities.vowelSwaps(reduction, $(vowelSwapLimit)))
  }

  def getSortedWordsByFrequency(words: List[String], input: String): List[(String, Long)] = {
    val filteredWords = words.filter(_.length >= input.length)
    val sortedWordsByFrequency = filteredWords.map(word => (word, compareFrequencies(word)))
      .sortBy(_._2).takeRight($(intersections))
    logger.debug(s"recommended by frequency: ${sortedWordsByFrequency.mkString(", ")}")
    sortedWordsByFrequency
  }

  def getSortedWordsByHamming(words: List[String], input: String): List[(String, Long)] = {
    val sortedWordByHamming = words.map(word => (word, compareHammers(input)(word)))
      .sortBy(_._2).takeRight($(intersections))
    logger.debug(s"recommended by hamming: ${sortedWordByHamming.mkString(", ")}")
    sortedWordByHamming
  }

  def getResult(wordsByFrequency: List[(String, Long)], wordsByHamming: List[(String, Long)], input: String):
  String = {
    val intersectWords = wordsByFrequency.map(word => word._1).intersect(wordsByHamming.map(word => word._1))
    if (wordsByFrequency.isEmpty && wordsByHamming.isEmpty) {
      logger.debug("no intersection or frequent words found")
      input
    } else if (wordsByFrequency.isEmpty || wordsByHamming.isEmpty) {
      logger.debug("no intersection but one recommendation found")
      (wordsByFrequency ++ wordsByHamming).last._1
    } else if (intersectWords.nonEmpty) {
      logger.debug("hammer and frequency recommendations found")
      val wordsByFrequencyAndHamming = intersectWords.map{word =>
        val frequency = wordsByFrequency.find(_._1 == word).get._2
        val hamming = wordsByHamming.find(_._1 == word).get._2
        (word, frequency * hamming)
      }
      wordsByFrequencyAndHamming.maxBy(_._2)._1
    } else {
      logger.debug("no intersection of hammer and frequency")
      Seq(wordsByFrequency.last._1, wordsByHamming.last._1).maxBy{word =>
        compareFrequencies(word) * compareHammers(input)(word)
      }
    }
  }

}

trait PretrainedNorvigSweeting {
  def pretrained(name: String = "spellcheck_norvig", lang: String = "en",
                 remoteLoc: String = ResourceDownloader.publicLoc): NorvigSweetingModel =
    ResourceDownloader.downloadModel(NorvigSweetingModel, name, Option(lang), remoteLoc)
}

object NorvigSweetingModel extends ParamsAndFeaturesReadable[NorvigSweetingModel] with PretrainedNorvigSweeting