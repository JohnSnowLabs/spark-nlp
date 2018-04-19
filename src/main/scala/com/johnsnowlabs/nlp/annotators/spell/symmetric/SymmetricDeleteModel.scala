package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.{HashSet, ListMap}
import scala.collection.mutable.{ListBuffer, Map => MMap} //MMap is a mutable object
import scala.util.control.Breaks._
import scala.math._

import org.apache.spark.ml.param.Param



class SymmetricDeleteModel(override val uid: String) extends AnnotatorModel[SymmetricDeleteModel] with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */
  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  private val alphabet = "abcdefghijjklmnopqrstuvwxyz".toCharArray
  private val vowels = "aeiouy".toCharArray

  // protected val wordCount: MapFeature[String, Long] = new MapFeature(this, "wordCount")
  protected val deriveWordCount: MapFeature[String, (ListBuffer[String], Long)] =
                                  new MapFeature(this, "deriveWordCount")
  protected val customDict: MapFeature[String, String] = new MapFeature(this, "customDict")

  val longestWordLength = new Param[Int](this, "longestWordLength", "length of longest word in corpus")

  def getLongestWordLength: Int = $(longestWordLength)

  def setLongestWordLength(value: Int): this.type = set(longestWordLength, value)

  private val logger = LoggerFactory.getLogger("NorvigApproach")
  private val config: Config = ConfigFactory.load

  /** params */
  private val wordSizeIgnore = config.getInt("nlp.norvigChecker.wordSizeIgnore")
  private val dupsLimit = config.getInt("nlp.norvigChecker.dupsLimit")
  private val reductLimit = config.getInt("nlp.norvigChecker.reductLimit")
  private val intersections = config.getInt("nlp.norvigChecker.intersections")
  private val vowelSwapLimit = config.getInt("nlp.norvigChecker.vowelSwapLimit")

  private lazy val allWords: HashSet[String] = {
    //HashSet($$(wordCount).keys.toSeq.map(_.toLowerCase):_*)
    HashSet($$(deriveWordCount).keys.toSeq.map(_.toLowerCase):_*)
  }

  def this() = this(Identifiable.randomUID("SPELL"))

  //def setWordCount(value: Map[String, Long]): this.type = set(wordCount, value)
  def setDeriveWordCount(value: Map[String, (ListBuffer[String], Long)]) :
                        this.type = set(deriveWordCount, value)
  def setCustomDict(value: Map[String, String]): this.type = set(customDict, value)

  //protected def getWordCount: Map[String, Long] = $$(wordCount)
  protected def getDeriveWordCount: Map[String, (ListBuffer[String], Long)] = $$(deriveWordCount)
  protected def getCustomDict: Map[String, String] = $$(customDict)

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
          if (dups < overrideLimit.getOrElse(dupsLimit)) {
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

  //private def compareFrequencies(value: String): Long = frequency(value, $$(wordCount))
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
  private def doubleVariants(word: String): Set[String] = {
    variants(word).flatMap(v => variants(v))
  }

  /** possible variations of the word by removing duplicate letters */
  /* ToDo: convert logic into an iterator, probably faster */
  private def reductions(word: String): Set[String] = {
    val flatWord: List[List[String]] = word.toCharArray.toList.zipWithIndex.collect {
      case (c, i) =>
        val n = numberOfDups(word, i)
        if (n > 0) {
          (0 to n).map(r => c.toString*r).take(reductLimit).toList
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
    if (word.length > vowelSwapLimit) return Set.empty[String]
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
    } else if ($$(customDict).contains(word)) {
      logger.debug("Word custom dictionary found. Replacing")
      Some($$(customDict)(word))
    } else if (allWords.contains(word.distinct)) {
      logger.debug("Word as distinct found in dictionary")
      Some(word.distinct)
    }  else if (word.length <= wordSizeIgnore) {
      logger.debug("word ignored because length is less than wordSizeIgnore")
      Some(word)
    } else None
  }

  private def suggestions(word: String): List[String] = {
    val intersectedPossibilities = allWords.intersect({
      val base =
        reductions(word) ++
          vowelSwaps(word) ++
          variants(word) ++
          both(word)
      base
    })
    if (intersectedPossibilities.nonEmpty) intersectedPossibilities.toList
    else List.empty[String]
  }

  /** Created by danilo 16/04/2018
    * Return list of suggested corrections for potentially incorrectly
    * spelled word
    * */
  def getSuggestedCorrections(string: String, silent: Boolean): (String, (Long, Int))  = {

    if ((string.length - this.getLongestWordLength) > $(maxEditDistance)){
      if (!silent){
        logger.debug("No items in dictionary within maximum edit distance")
      }
      ()
    }

    var minSuggestLen: Double = Double.PositiveInfinity

    val suggestDict = MMap[String, (Long, Int)]()
    val queueDictionary = MMap[String, String]() // items other than string that we've checked
    var queueList = ListBuffer(string)

    while (queueList.nonEmpty){
      val queueItem = queueList.head // pop
      queueList = queueList.slice(1, queueList.length)

      breakable{ //early exit
        if (suggestDict.nonEmpty && (string.length - queueItem.length) > $(maxEditDistance)){
          break
        }
      }

      // process queue item
      if (allWords.contains(queueItem) && suggestDict.contains(queueItem)) {

        val dictionary = $$(deriveWordCount)

        if (dictionary(queueItem)._2 > 0) {
          // word is in dictionary, and is a word from the corpus, and not already in suggestion list
          // so add to suggestion dictionary, indexed by the word with value:
          // (frequency in corpus, edit distance)
          // note q_items that are not the input string are shorter than input string since only
          // deletes are added (unless manual dictionary corrections are added)
          suggestDict(queueItem) = (dictionary(queueItem)._2,
                                    string.length - queueItem.length)

          breakable{ //early exit
            if (string.length == queueItem.length){
              break
            }
          }

          if (string.length - queueItem.length < minSuggestLen){
            minSuggestLen = string.length - queueItem.length
          }
        }

        // the suggested corrections for q_item as stored in dictionary (whether or not queueItem itself
        // is a valid word or merely a delete) can be valid corrections
        dictionary(queueItem)._1.foreach( scItem => {
          if (!suggestDict.contains(scItem)){
            // assert(scItem.length > queueItem.length) Include or not assertions ???

            // calculate edit distance using Damerau-Levenshtein distance
            val itemDist = levenshteinDistance(scItem, string)

            if (itemDist <= $(maxEditDistance)){
              suggestDict(scItem) = (dictionary(scItem)._2, itemDist)
              if (itemDist < minSuggestLen) {
                minSuggestLen = itemDist
              }
            }
            // depending on order words are processed, some words with different edit distances may be
            // entered into suggestions; trim suggestion dictionary
            suggestDict.retain((k, v) => v._2 <= minSuggestLen)
          }
        })

      }

      // now generate deletes (e.g. a substring of string or of a delete) from the queue item
      // do not add words with greater edit distance
      if ((string.length - queueItem.length) < $(maxEditDistance) && queueItem.length > 1){
        val y = 0 until queueItem.length
        y.foreach(c => { //character index
          //result of word minus c
          val wordMinus = queueItem.substring(0, c).concat(queueItem.substring(c+1, queueItem.length))
          if (!queueDictionary.contains(wordMinus)){
            queueList += wordMinus
            queueDictionary(wordMinus) = "None" // arbitrary value, just to identify we checked this
          }
        }) // End queueItem.foreach
      }

    } // End while

    // return list of suggestions with (correction, (frequency in corpus, edit distance))
    var suggestions = ListMap(suggestDict.toSeq.sortBy(_._2):_*).toList

    if (suggestions.isEmpty){
      suggestions = List((string, (0, 0)))
    }

    suggestions.head
  }

  private def minimum(i1: Int, i2: Int, i3: Int)=min(min(i1, i2), i3)

  def levenshteinDistance(s1:String, s2:String): Int ={
    val dist=Array.tabulate(s2.length+1, s1.length+1){(j,i)=>if(j==0) i else if (i==0) j else 0}

    for(j<-1 to s2.length; i<-1 to s1.length)
      dist(j)(i)=if(s2(j-1)==s1(i-1)) dist(j-1)(i-1)
      else minimum(dist(j-1)(i)+1, dist(j)(i-1)+1, dist(j-1)(i-1)+1)

    dist(s2.length)(s1.length)
  }

  def check(raw: String): String = {
    logger.debug(s"spell checker target word: $raw")
    val silent = true
    val word = getSuggestedCorrections(raw, silent)

    logger.debug(s"Received: $raw. Best correction is: $word. " +
      s"Because frequency was ${word._2._1} " +
      s"and edit distance was ${word._2._2}")
    word._1
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
  def pretrained(name: String = "spell_fast", folder: String = "", language: Option[String] = Some("en")): SymmetricDeleteModel =
    ResourceDownloader.downloadModel(SymmetricDeleteModel, name, folder, language)
}

object SymmetricDeleteModel extends ParamsAndFeaturesReadable[SymmetricDeleteModel] with PretrainedNorvigSweeting
