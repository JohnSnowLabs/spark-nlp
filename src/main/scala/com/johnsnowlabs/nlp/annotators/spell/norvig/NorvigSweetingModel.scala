package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashSet

class NorvigSweetingModel(override val uid: String) extends AnnotatorModel[NorvigSweetingModel] with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */
  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  private val alphabet = "abcdefghijjklmnopqrstuvwxyz".toCharArray
  private val vowels = "aeiouy".toCharArray

  protected val wordCount: MapFeature[String, Long] = new MapFeature(this, "wordCount")
  //protected val customDict: MapFeature[String, String] = new MapFeature(this, "customDict")

  private val logger = LoggerFactory.getLogger("NorvigApproach")

  /** params */
  protected val wordSizeIgnore = new IntParam(this, "wordSizeIgnore", "minimum size of word before ignoring. Defaults to 3")
  protected val dupsLimit = new IntParam(this, "dupsLimit", "maximum duplicate of characters in a word to consider. Defaults to 2")
  protected val reductLimit = new IntParam(this, "reductLimit", "word reductions limit. Defaults to 3")
  protected val intersections = new IntParam(this, "intersections", "hamming intersections to attempt. Defaults to 10")
  protected val vowelSwapLimit = new IntParam(this, "vowelSwapLimit", "vowel swap attempts. Defaults to 6")

  def setWordSizeIgnore(v: Int) = set(wordSizeIgnore, v)
  def setDupsLimit(v: Int) = set(dupsLimit, v)
  def setReductLimit(v: Int) = set(reductLimit, v)
  def setIntersections(v: Int) = set(intersections, v)
  def setVowelSwapLimit(v: Int) = set(vowelSwapLimit, v)

  private lazy val allWords: HashSet[String] = {
    if ($(caseSensitive)) HashSet($$(wordCount).keys.toSeq:_*) else HashSet($$(wordCount).keys.toSeq.map(_.toLowerCase):_*)
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  def setWordCount(value: Map[String, Long]): this.type = set(wordCount, value)
  //def setCustomDict(value: Map[String, String]): this.type = set(customDict, value)

  protected def getWordCount: Map[String, Long] = $$(wordCount)
  //protected def getCustomDict: Map[String, String] = $$(customDict)

  /** Utilities */
  /** number of items duplicated in some text */
  def cartesianProduct[T](xss: List[List[_]]): List[List[_]] = xss match {
    case Nil => List(Nil)
    case h :: t => for (xh <- h; xt <- cartesianProduct(t)) yield xh :: xt
  }

  private def numberOfDups(text: String, id: Int): Int = {
    var idx = id
    val initialId = idx
    val last = text(idx)
    while (idx+1 < text.length && text(idx+1) == last) {
      idx += 1
    }
    idx - initialId
  }

  private def limitDups(text: String, overrideLimit: Option[Int] = None): String = {
    var dups = 0
    text.zipWithIndex.collect {
      case (w, i) =>
        if (i == 0) {
          w
        }
        else if (w == text(i - 1)) {
          if (dups < overrideLimit.getOrElse($(dupsLimit))) {
            dups += 1
            w
          } else {
            ""
          }
        } else {
          dups = 0
          w
        }
    }.mkString("")
  }

  /** distance measure between two words */
  private def hammingDistance(word1: String, word2: String): Int =
    if (word1 == word2) 0
    else word1.zip(word2).count { case (c1, c2) => c1 != c2 } + (word1.length - word2.length).abs

  /** retrieve frequency */
  private def frequency(word: String, wordCount: Map[String, Long]): Long = {
    wordCount.getOrElse(word, 0)
  }

  private def compareFrequencies(value: String): Long = frequency(value, $$(wordCount))
  private def compareHammers(input: String)(value: String): Int = hammingDistance(input, value)

  /** Posibilities analysis */
  private def variants(targetWord: String): Set[String] = {
    val splits = (0 to targetWord.length).map(i => (targetWord.take(i), targetWord.drop(i)))
    val deletes = splits.collect {
      case (a,b) if b.length > 0 => a + b.tail
    }
    val transposes = splits.collect {
      case (a,b) if b.length > 1 => a + b(1) + b(0) + b.drop(2)
    }
    val replaces = splits.collect {
      case (a, b) if b.length > 0 => alphabet.map(c => a + c + b.tail)
    }.flatten
    val inserts = splits.collect {
      case (a, b) => alphabet.map(c => a + c + b)
    }.flatten
    val vars = Set(deletes ++ transposes ++ replaces ++ inserts :_ *)
    logger.debug("variants proposed: " + vars.size)
    vars
  }

  /** variants of variants of a word */
  private def doubleVariants(word: String): Set[String] =
    variants(word).flatMap(v => variants(v))

  /** possible variations of the word by removing duplicate letters */
  /* ToDo: convert logic into an iterator, probably faster */
  private def reductions(word: String): Set[String] = {
    val flatWord: List[List[String]] = word.toCharArray.toList.zipWithIndex.collect {
      case (c, i) =>
        val n = numberOfDups(word, i)
        if (n > 0) {
          (0 to n).map(r => c.toString*r).take($(reductLimit)).toList
        } else {
          List(c.toString)
        }
    }
    val reds = cartesianProduct(flatWord).map(_.mkString("")).toSet
    logger.debug("parsed reductions: " + reds.size)
    reds
  }

  /** flattens vowel possibilities */
  private def vowelSwaps(word: String): Set[String] = {
    if (word.length > $(vowelSwapLimit)) return Set.empty[String]
    val flatWord: List[List[Char]] = word.toCharArray.collect {
      case c => if (vowels.contains(c)) {
        vowels.toList
      } else {
        List(c)
      }
    }.toList
    val vswaps = cartesianProduct(flatWord).map(_.mkString("")).toSet
    logger.debug("vowel swaps: " + vswaps.size)
    vswaps
  }

  private def both(word: String): Set[String] = {
    reductions(word).flatMap(vowelSwaps)
  }

  /** get best spelling suggestion */
  private def suggestion(word: String): Option[String] = {
    if (allWords.contains(word)) {
      logger.debug("Word found in dictionary. No spell change")
      Some(word)
    /*} else if ($$(customDict).contains(word)) {
      logger.debug("Word custom dictionary found. Replacing")
      Some($$(customDict)(word))*/
    } else if (word.length <= $(wordSizeIgnore)) {
      logger.debug("word ignored because length is less than wordSizeIgnore")
      Some(word)
    } else if (allWords.contains(word.distinct)) {
      logger.debug("Word as distinct found in dictionary")
      Some(word.distinct)
    } else if ($(shortCircuit)) {
      if (allWords.intersect(reductions(word)).nonEmpty) Some(word)
      else if (allWords.intersect(vowelSwaps(word)).nonEmpty) Some(word)
      else if (allWords.intersect(variants(word)).nonEmpty) Some(word)
      else if (allWords.intersect(both(word)).nonEmpty) Some(word)
      else if ($(doubleVariants) && allWords.intersect(doubleVariants(word)).nonEmpty) Some(word)
      else None
    } else None
  }

  private def suggestions(word: String): List[String] = {
    val intersectedPossibilities = allWords.intersect({
      val base =
        reductions(word) ++
          vowelSwaps(word) ++
          variants(word) ++
          both(word)
      if ($(doubleVariants)) base ++ doubleVariants(word) else base
    })
    if (intersectedPossibilities.nonEmpty) intersectedPossibilities.toList
    else List.empty[String]
  }

  def check(raw: String): String = {
    val input = limitDups(raw)
    logger.debug(s"spell checker target word: $input")
    val possibility = suggestion(input)
    if (possibility.isDefined) return possibility.get
    val listedSuggestions = suggestions(input)
    val sortedFreq = listedSuggestions.filter(_.length >= input.length).sortBy(compareFrequencies).takeRight($(intersections))
    logger.debug(s"recommended by frequency: ${sortedFreq.mkString(", ")}")
    val sortedHamm = listedSuggestions.sortBy(compareHammers(input)).takeRight($(intersections))
    logger.debug(s"recommended by hamming: ${sortedHamm.mkString(", ")}")
    val intersect = sortedFreq.intersect(sortedHamm)
    /* Picking algorithm */
    val result =
      if (sortedFreq.isEmpty && sortedHamm.isEmpty) {
        logger.debug("no intersection or frequent words found")
        input
      } else if (sortedFreq.isEmpty || sortedHamm.isEmpty) {
        logger.debug("no intersection but one recommendation found")
        (sortedFreq ++ sortedHamm).last
      } else if (intersect.nonEmpty) {
        logger.debug("hammer and frequency recommendations found")
        intersect.last
      } else {
        logger.debug("no intersection of hammer and frequency")
        Seq(sortedFreq.last, sortedHamm.last).maxBy(w => compareFrequencies(w) * compareHammers(input)(w))
      }
    logger.debug(s"Received: $input. Best correction is: $result. " +
      s"Because frequency was ${compareFrequencies(result)} " +
      s"and hammer score is ${compareHammers(input)(result)}")
    result
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token =>
        Annotation(
          annotatorType,
          token.begin,
          token.end,
          check(token.result),
          token.metadata
        )
    }
  }
}

trait PretrainedNorvigSweeting {
  def pretrained(name: String = "spell_fast", language: Option[String] = Some("en"),
                 remoteLoc: String = ResourceDownloader.publicLoc): NorvigSweetingModel =
    ResourceDownloader.downloadModel(NorvigSweetingModel, name, language, remoteLoc)
}

object NorvigSweetingModel extends ParamsAndFeaturesReadable[NorvigSweetingModel] with PretrainedNorvigSweeting