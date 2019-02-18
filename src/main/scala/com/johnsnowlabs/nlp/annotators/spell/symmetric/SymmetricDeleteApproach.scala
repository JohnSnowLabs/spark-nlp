package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
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
  val dictionary = new ExternalResourceParam(this, "dictionary", "file with a list of correct words")

  setDefault(maxEditDistance, 3)

  def setCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "spell checker corpus needs 'tokenPattern' regex for " +
                                                    "tagging words. e.g. [a-zA-Z]+")
    set(corpus, value)
  }

  def setCorpus(path: String,
                tokenPattern: String = "\\S+",
                readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                options: Map[String, String] = Map("format" -> "text")): this.type =
    set(corpus, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  def setDictionary(path: String,
                    tokenPattern: String = "\\S+",
                    readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                    options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))


  // AnnotatorType shows the structure of the result, we can have annotators with the same result
  override val annotatorType: AnnotatorType = TOKEN

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

  /** Created by danilo 26/04/2018
    * Computes derived words from a frequency of words
    * */
  def derivedWordDistances(wordFrequencies: List[(String, Long)], maxEditDistance: Int): WordFeatures = {

    val derivedWords = collection.mutable.Map() ++ wordFrequencies.map{a => (a._1, (ListBuffer[String](),a._2))}.toMap
    val wordFeatures = WordFeatures(derivedWords, 0)
    wordFrequencies.foreach(word =>{

      val deletes = getDeletes(word._1, maxEditDistance)

      deletes.foreach( item => {
        if (derivedWords.contains(item)){
          // add (correct) word to delete's suggested correction list
          derivedWords(item)._1 += word._1
        } else {
          // note frequency of word in corpus is not incremented
          val wordFrequency = new ListBuffer[String]
          wordFrequency += word._1
          derivedWords(item) = (wordFrequency, 0)
        }
      }) // End deletes.foreach
    })
    wordFeatures.derivedWords = derivedWords
    wordFeatures
  }

  case class WordFeatures(var derivedWords: MMap[String, (ListBuffer[String], Long)],
                          var longestWordLength: Int)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    val possibleDict = get(dictionary).map(d => ResourceHelper.wordCount(d))
    var wordFeatures = WordFeatures(MMap.empty, 0)

    if (get(corpus).isDefined) {

      val corpusWordCount = ResourceHelper.wordCount($(corpus), p = recursivePipeline).toMap
      val wordFrequencies = corpusWordCount.flatMap{case (word, wordFrequency) =>
        List((word, wordFrequency))}
      wordFeatures = derivedWordDistances(wordFrequencies.toList, $(maxEditDistance))
      val longestWord = wordFrequencies.keysIterator.reduceLeft((word, nextWord) =>
        if (word.length > nextWord.length) word else nextWord)
      wordFeatures.longestWordLength = longestWord.length

    } else {
      import ResourceHelper.spark.implicits._
      import org.apache.spark.sql.functions._

      require(!dataset.rdd.isEmpty(), "corpus not provided and dataset for training is empty")

      val trainDataset = dataset.select($(inputCols).head).as[Array[Annotation]]
                        .flatMap(_.map(_.result))
      val wordFrequencies = trainDataset.groupBy("value").count().as[(String, Long)].collect.toList
      wordFeatures = derivedWordDistances(wordFrequencies, $(maxEditDistance))
      wordFeatures.longestWordLength = trainDataset.agg(max(length(col("value")))).head().getInt(0)
    }

    val model = new SymmetricDeleteModel()
      .setDerivedWords(wordFeatures.derivedWords.mapValues(derivedWords =>
        (derivedWords._1.toList, derivedWords._2)).toMap)
      .setLongestWordLength(wordFeatures.longestWordLength)

    if (possibleDict.isDefined)
      model.setDictionary(possibleDict.get.toMap)

    model
  }

}
// This objects reads the class' properties, it enables reading the model after it is stored
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
