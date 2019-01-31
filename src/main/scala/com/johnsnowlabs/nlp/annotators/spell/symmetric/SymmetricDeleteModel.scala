package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.spell.common.LevenshteinDistance
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashSet
import scala.collection.mutable.{Map => MMap}
import scala.util.control.Breaks._
import scala.math._
import org.apache.spark.ml.param.IntParam

/** Created by danilo 16/04/2018,
  * inspired on https://github.com/wolfgarbe/SymSpell
  *
  * The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
  * dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
  * (than the standard approach with deletes + transposes + replaces + inserts) and language independent.
  * */
class SymmetricDeleteModel(override val uid: String) extends AnnotatorModel[SymmetricDeleteModel]
  with SymmetricDeleteParams with LevenshteinDistance {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */
  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  protected val derivedWords: MapFeature[String, (List[String], Long)] =
    new MapFeature(this, "derivedWords")

  protected val dictionary: MapFeature[String, Long] = new MapFeature(this, "dictionary")

  val longestWordLength = new IntParam(this, "longestWordLength",
                                "length of longest word in corpus")

  def getLongestWordLength: Int = $(longestWordLength)

  def setLongestWordLength(value: Int): this.type = set(longestWordLength, value)

  def setDictionary(value: Map[String, Long]) = set(dictionary, value)

  private val logger = LoggerFactory.getLogger("SymmetricDeleteApproach")

  private lazy val allWords: HashSet[String] = {
    HashSet($$(derivedWords).keys.toSeq.map(_.toLowerCase): _*)
  }

  private val CAPITAL = 'C'
  private val LOWERCASE = 'L'
  private val UPPERCASE = 'U'

  def this() = this(Identifiable.randomUID("SYMSPELL"))

  def setDerivedWords(value: Map[String, (List[String], Long)]):
  this.type = set(derivedWords, value)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.map { token => {
      Annotation(
        annotatorType,
        token.begin,
        token.end,
        check(token.result).getOrElse(token.result),
        token.metadata
      )
    }}
  }

  def check(originalWord: String): Option[String] = {
    logger.debug(s"spell checker target word: $originalWord")

    if (isNoisyWord(originalWord)) {
      return Option(originalWord)
    }
    var transformedWord = originalWord
    val originalCaseType = getCaseWordType(originalWord)
    val correctedWord = getSuggestedCorrections(originalWord)
    if (correctedWord.isDefined) {
      logger.debug(s"Received: $originalWord. Best correction is: $correctedWord. " +
        s"Because frequency was ${correctedWord.get._2._1} " +
        s"and edit distance was ${correctedWord.get._2._2}")
      transformedWord = transformToOriginalCaseType(originalCaseType, correctedWord.map(_._1).getOrElse(""))
    }

    Option(transformedWord)
  }

  def isNoisyWord(word: String): Boolean = {
    val noisyWordRegex = "[^a-zA-Z]".r
    val matchNoisyWord = noisyWordRegex.findFirstMatchIn(word)

    if (matchNoisyWord.isEmpty){
      false
    } else {
      true
    }
  }

  def getCaseWordType(word: String): Char = {
    val firstLetter = word(0).toString
    val matchUpperCaseFirstLetter = "[A-Z]".r.findFirstMatchIn(firstLetter)

    var caseType = UPPERCASE

    word.foreach{letter =>
      val matchUpperCase = "[A-Z]".r.findFirstMatchIn(letter.toString)
      if (matchUpperCase.isEmpty){
        if (matchUpperCaseFirstLetter.nonEmpty) {
          caseType = CAPITAL
        } else {
          caseType = LOWERCASE
        }
      }
    }

    caseType
  }

  def transformToOriginalCaseType(caseType: Char, word: String): String ={

    var transformedWord = word

    if (caseType == CAPITAL){
      val firstLetter = word(0).toString
      transformedWord = word.replaceFirst(firstLetter, firstLetter.toUpperCase)
    } else if(caseType == UPPERCASE) {
      transformedWord = word.toUpperCase
    }
    transformedWord
  }

  /** Return list of suggested corrections for potentially incorrectly
    * spelled word
    * */

  def getSuggestedCorrections(word: String): Option[(String, (Long, Int))] = {
    val lowercaseWord = word.toLowerCase()
    val lowercaseWordLength = lowercaseWord.length
    if ((get(dictionary).isDefined && $$(dictionary).contains(word)) || ((lowercaseWordLength - this.getLongestWordLength) > $(maxEditDistance)))
      return None

    var minSuggestLen: Double = Double.PositiveInfinity

    val suggestDict = MMap.empty[String, (Long, Int)]
    val queueDictionary = MMap.empty[String, String] // items other than string that we've checked
    var queueList = Iterator(lowercaseWord)

    while (queueList.hasNext) {
      val queueItem = queueList.next // pop
      val queueItemLength = queueItem.length

      breakable { //early exit
        if (suggestDict.nonEmpty && (lowercaseWordLength - queueItemLength) > $(maxEditDistance)) {
          break
        }
      }

      // process queue item
      if (allWords.contains(queueItem) && !suggestDict.contains(queueItem)) {

//        if (queueItem == "cotde"){
//          println("debug...")
//          val suggestionList = $$(derivedWords).getOrElse(queueItem, (List(""), 0))
//          println(value)
//        }

        var suggestedWordsWeight: (List[String], Long) = $$(derivedWords).getOrElse(queueItem, (List(""), 0))

        if (suggestedWordsWeight._2 > 0) {
          // word is in dictionary, and is a word from the corpus, and not already in suggestion list
          // so add to suggestion dictionary, indexed by the word with value:
          // (frequency in corpus, edit distance)
          // note q_items that are not the input string are shorter than input string since only
          // deletes are added (unless manual dictionary corrections are added)
          suggestDict(queueItem) = (suggestedWordsWeight._2,
            lowercaseWordLength - queueItemLength)

          breakable { //early exit
            if (lowercaseWordLength == queueItemLength) {
              break
            }
          }

          if (lowercaseWordLength - queueItemLength < minSuggestLen) {
            minSuggestLen = lowercaseWordLength - queueItemLength
          }
        }

        // the suggested corrections for q_item as stored in dictionary (whether or not queueItem itself
        // is a valid word or merely a delete) can be valid corrections
        suggestedWordsWeight._1.foreach(scItem => {
          val lowercaseScItem = scItem.toLowerCase
          if (!suggestDict.contains(lowercaseScItem) && lowercaseScItem != "") {

            // calculate edit distance using Damerau-Levenshtein distance
            val itemDist = levenshteinDistance(lowercaseScItem, lowercaseWord)

            if (itemDist <= $(maxEditDistance)) {
              suggestedWordsWeight = $$(derivedWords).getOrElse(lowercaseScItem, (List(""), 0))
              if (suggestedWordsWeight._2 > 0){
                suggestDict(lowercaseScItem) = (suggestedWordsWeight._2,
                  itemDist)
                if (itemDist < minSuggestLen) {
                  minSuggestLen = itemDist
                }
              }
            }
            // depending on order words are processed, some words with different edit distances may be
            // entered into suggestions; trim suggestion dictionary
            suggestDict.retain((_, v) => v._2 <= minSuggestLen)
          }
        })

      }

      // now generate deletes (e.g. a substring of string or of a delete) from the queue item
      // do not add words with greater edit distance
      if ((lowercaseWordLength - queueItemLength) < $(maxEditDistance) && queueItemLength > 1) {
        val y = 0 until queueItemLength
        y.foreach(c => { //character index
          //result of word minus c
          val wordMinus = queueItem.substring(0, c).concat(queueItem.substring(c + 1, queueItemLength))
          if (!queueDictionary.contains(wordMinus)) {
            queueList ++= Iterator(wordMinus)
            queueDictionary(wordMinus) = "None" // arbitrary value, just to identify we checked this
          }
        }) // End queueItem.foreach
      }

    } // End while

    // return list of suggestions with (correction, (frequency in corpus, edit distance))

    val suggestions = suggestDict.toSeq.sortBy { case (k, (f, d)) => (d, -f, k) }.toList
    suggestions.headOption.orElse(None)

  }

}

trait PretrainedSymmetricDelete {
  def pretrained(name: String = "spell_sd_fast", language: Option[String] = Some("en"),
                 remoteLoc: String = ResourceDownloader.publicLoc): SymmetricDeleteModel =
    ResourceDownloader.downloadModel(SymmetricDeleteModel, name, language, remoteLoc)
}

object SymmetricDeleteModel extends ParamsAndFeaturesReadable[SymmetricDeleteModel] with PretrainedSymmetricDelete
