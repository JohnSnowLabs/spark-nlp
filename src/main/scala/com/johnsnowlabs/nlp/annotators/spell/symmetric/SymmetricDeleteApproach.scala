package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.{Map => MMap}

/** Created by danilo 16/04/2018,
  * Symmetric Delete spelling correction algorithm
  * inspired on https://github.com/wolfgarbe/SymSpell
  * */
class SymmetricDeleteApproach(override val uid: String)
  extends AnnotatorApproach[SymmetricDeleteModel]
    with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Spell checking algorithm inspired on Symmetric Delete algorithm"

  val corpus = new ExternalResourceParam(this, "corpus", "folder or file with text that teaches about the language")

  setDefault(maxEditDistance, 3)
  //setDefault(longestWordLength, 0)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "spell checker corpus needs 'tokenPattern' regex for " +
                                                    "tagging words. e.g. [a-zA-Z]+")
    set(corpus, value)
  }

  def setCorpus(path: String,
                tokenPattern: String,
                readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))


  // AnnotatorType shows the structure of the result, we can have annotators with the same result
  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN) //The approach required to work

  def this() = this(Identifiable.randomUID("SYMSPELL")) // constructor required for the annotator to work in python

  /** Created by danilo 14/04/2018
    * Given a word, derive strings with up to maxEditDistance characters
    * deleted
    * */
  def getDeletes(word: String, med: Int): List[String] ={

    var deletes = new ListBuffer[String]()
    var queueList = List(word)
    val x = 1 to med
    x.foreach( d =>
    {
      var tempQueue = new ListBuffer[String]()
      queueList.foreach(w => {
        if (w.length > 1){
          val y = 0 until w.length
          y.foreach(c => { //character index
            //result of word minus c
            val wordMinus = w.substring(0, c).concat(w.substring(c+1, w.length))
            if (!deletes.contains(wordMinus)){
              deletes += wordMinus
            }
            if (!tempQueue.contains(wordMinus)){
              tempQueue += wordMinus
            }
          }) // End y.foreach
          queueList = tempQueue.toList
        }
      }
      ) //End queueList.foreach
    }
    ) //End x.foreach

    deletes.toList
  }

  /** Created by danilo 17/04/2018
    * check if word is already in dictionary
    * dictionary entries are in the form:
    * (list of suggested corrections,frequency of word in corpus)
    * */
  def updateDictionary(d: MMap[String, (ListBuffer[String], Long)],
                       word: String, maxEditDistance: Int, longestWordLength: Int
                      ): Int = {

    var newLongestWordLength = longestWordLength

    if (d(word)._2 == 0) {
      d(word) = (ListBuffer[String](), 1)
      newLongestWordLength = word.length.max(longestWordLength)
    } else{
      var count: Long = d(word)._2
      // increment count of word in corpus
      count += 1
      d(word) = (d(word)._1, count)
    }

    if (d(word)._2 == 1){
      val deletes = getDeletes(word, maxEditDistance)

      deletes.foreach( item => {
        if (d.contains(item)){
          // add (correct) word to delete's suggested correction list
          d(item)._1 += word
        } else {
          // note frequency of word in corpus is not incremented
          val wordFrequency = new ListBuffer[String]
          wordFrequency += word
          d(item) = (wordFrequency, 0)
        }
      }) // End deletes.foreach
    }
    newLongestWordLength
  }

  def derivedWordDistances(externalResource: List[String],
                           derivedWords: MMap[String, (ListBuffer[String], Long)] =
                           MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0),
                           maxEditDistance: Int
                          ): WordsFeatures = {
    val regex = $(corpus).options("tokenPattern").r
    val wordFeatures = WordsFeatures(MMap(), 0)
    var longestWordLength = wordFeatures.longestWordLength
    externalResource.foreach(line => {
      val tokenizeWords = regex.findAllMatchIn(line).map(_.matched).toList
      tokenizeWords.foreach(word => {
        longestWordLength = updateDictionary(derivedWords, word, maxEditDistance, longestWordLength)
      })
    })
    wordFeatures.longestWordLength = longestWordLength
    wordFeatures.derivedWords = derivedWords
    wordFeatures
  }

  case class WordsFeatures(var derivedWords: MMap[String, (ListBuffer[String], Long)] =
                           MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0),
                           var longestWordLength: Int)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    val externalResource = ResourceHelper.getExternalResourceAsList($(corpus))

    val wordFeatures = derivedWordDistances(externalResource = externalResource,
                                            maxEditDistance = $(maxEditDistance))

    new SymmetricDeleteModel()
      .setDictionary(wordFeatures.derivedWords.toMap)
      .setLongestWordLength(wordFeatures.longestWordLength)
  }

}
// This objects reads the class' properties, it enables reding the model after it is stored
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
