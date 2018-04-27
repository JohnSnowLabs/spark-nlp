package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.util.Identifiable
import org.slf4j.LoggerFactory

import scala.collection.immutable.HashSet
import scala.collection.mutable.{ListBuffer, Map => MMap} //MMap is a mutable object
import scala.util.control.Breaks._
import scala.math._

import org.apache.spark.ml.param.Param


/** Created by danilo 16/04/2018,
  * inspired on https://github.com/wolfgarbe/SymSpell
  *
  * The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
  * dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
  * (than the standard approach with deletes + transposes + replaces + inserts) and language independent.
  * */
class SymmetricDeleteModel(override val uid: String) extends AnnotatorModel[SymmetricDeleteModel] with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /**
    * Annotator reference id. Used to identify elements in metadata or to refer to this annotator type
    */
  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  protected val derivedWords: MapFeature[String, (ListBuffer[String], Long)] =
                                  new MapFeature(this, "derivedWords")

  val longestWordLength = new Param[Int](this, "longestWordLength", "length of longest word in corpus")

  def getLongestWordLength: Int = $(longestWordLength)

  def setLongestWordLength(value: Int): this.type = set(longestWordLength, value)

  private val logger = LoggerFactory.getLogger("SymmetricDeleteApproach")

  private lazy val allWords: HashSet[String] = {
    //HashSet($$(wordCount).keys.toSeq.map(_.toLowerCase):_*)
    HashSet($$(derivedWords).keys.toSeq.map(_.toLowerCase):_*)
  }

  def this() = this(Identifiable.randomUID("SYMSPELL"))

  def setDerivedWords(value: Map[String, (ListBuffer[String], Long)]) :
                        this.type = set(derivedWords, value)


  /** Utilities */
  /** Computes Levenshtein distance :
    * Metric of measuring difference between two sequences (edit distance)
    * Source: https://rosettacode.org/wiki/Levenshtein_distance
    * */
  def levenshteinDistance(s1:String, s2:String): Int ={
    val dist=Array.tabulate(s2.length+1, s1.length+1){(j,i)=>if(j==0) i else if (i==0) j else 0}

    for(j<-1 to s2.length; i<-1 to s1.length)
      dist(j)(i)=if(s2(j-1)==s1(i-1)) dist(j-1)(i-1)
      else minimum(dist(j-1)(i)+1, dist(j)(i-1)+1, dist(j-1)(i-1)+1)

    dist(s2.length)(s1.length)
  }

  private def minimum(i1: Int, i2: Int, i3: Int)=min(min(i1, i2), i3)

  /** Return list of suggested corrections for potentially incorrectly
    * spelled word
    * */
  def getSuggestedCorrections(word: String, silent: Boolean): (String, (Long, Int))  = {
    val string = word.toLowerCase()
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
    var count = 0

    while (queueList.nonEmpty){
      count += 1
      val queueItem = queueList.head // pop
      queueList = queueList.slice(1, queueList.length)

      breakable{ //early exit
        if (suggestDict.nonEmpty && (string.length - queueItem.length) > $(maxEditDistance)){
          break
        }
      }

      // process queue item
      if (allWords.contains(queueItem) && !suggestDict.contains(queueItem)) {

        if ($$(derivedWords)(queueItem)._2 > 0) {
          // word is in dictionary, and is a word from the corpus, and not already in suggestion list
          // so add to suggestion dictionary, indexed by the word with value:
          // (frequency in corpus, edit distance)
          // note q_items that are not the input string are shorter than input string since only
          // deletes are added (unless manual dictionary corrections are added)
          suggestDict(queueItem) = ($$(derivedWords)(queueItem)._2,
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
        $$(derivedWords)(queueItem)._1.foreach( scItem => {
          if (!suggestDict.contains(scItem.toLowerCase())){
            // assert(scItem.length > queueItem.length) Include or not assertions ???

            // calculate edit distance using Damerau-Levenshtein distance
            val itemDist = levenshteinDistance(scItem.toLowerCase, string)

            if (itemDist <= $(maxEditDistance)){
              suggestDict(scItem.toLowerCase) = ($$(derivedWords)(scItem.toLowerCase)._2,
                                                itemDist)
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
    var suggestions = suggestDict.toSeq.sortBy {case (k,(f,d)) => (d,-f,k)}.toList

    if (suggestions.isEmpty){
      suggestions = List((string, (0, 0)))
    }

    suggestions.head
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

trait PretrainedSymmetricDelete { // ask if the name spell_sd_fast it's ok
  def pretrained(name: String = "spell_sd_fast", folder: String = "",
                 language: Option[String] = Some("en")): SymmetricDeleteModel =
    ResourceDownloader.downloadModel(SymmetricDeleteModel, name, folder, language)
}

object SymmetricDeleteModel extends ParamsAndFeaturesReadable[SymmetricDeleteModel] with PretrainedSymmetricDelete
