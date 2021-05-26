package com.johnsnowlabs.nlp.annotators.spell.symmetric

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{AnalysisException, Dataset}

import scala.collection.mutable.ListBuffer

/** Created by danilo 16/04/2018,
  * Symmetric Delete spelling correction algorithm. It retrieves tokens and utilizes distance metrics to compute possible derived words.
  *
  * Inspired by [[https://github.com/wolfgarbe/SymSpell]]
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/symmetric/SymmetricDeleteModelTestSpec.scala]] for further reference.
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  **/
class SymmetricDeleteApproach(override val uid: String)
  extends AnnotatorApproach[SymmetricDeleteModel]
    with SymmetricDeleteParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Spell checking algorithm inspired on Symmetric Delete algorith */
  override val description: String = "Spell checking algorithm inspired on Symmetric Delete algorithm"
  /** file with a list of correct words
    *
    * @group param
    **/
  val dictionary = new ExternalResourceParam(this, "dictionary", "file with a list of correct words")

  setDefault(
    frequencyThreshold -> 0,
    deletesThreshold -> 0,
    maxEditDistance -> 3,
    dupsLimit -> 2
  )

  /** Optional dictionary of properly written words. If provided, significantly boosts spell checking performance
    *
    * @group setParam
    **/
  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  /** Optional dictionary of properly written words. If provided, significantly boosts spell checking performance
    *
    * @group setParam
    **/
  def setDictionary(path: String,
                    tokenPattern: String = "\\S+",
                    readAs: ReadAs.Format = ReadAs.TEXT,
                    options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))


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

  def this() = this(Identifiable.randomUID("SYMSPELL")) // constructor required for the annotator to work in python

  /** Created by danilo 14/04/2018
    * Given a word, derive strings with up to maxEditDistance characters
    * deleted
    * */
  def getDeletes(word: String, med: Int): List[String] = {

    var deletes = new ListBuffer[String]()
    var queueList = List(word)
    val x = 1 to med
    x.foreach( _ =>
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
  def derivedWordDistances(wordFrequencies: List[(String, Long)], maxEditDistance: Int): Map[String, (List[String], Long)] = {

    val derivedWords = scala.collection.mutable.Map(wordFrequencies.map{a => (a._1, (ListBuffer.empty[String], a._2))}:_*)

    wordFrequencies.foreach{case (word, _) =>

      val deletes = getDeletes(word, maxEditDistance)

      deletes.foreach( deleteItem => {
        if (derivedWords.contains(deleteItem)){
          // add (correct) word to delete's suggested correction list
          derivedWords(deleteItem)._1 += word
        } else {
          // note frequency of word in corpus is not incremented
          derivedWords(deleteItem) = (ListBuffer(word), 0L)
        }
      }) // End deletes.foreach
    }
    derivedWords
      .filterKeys(a => derivedWords(a)._1.length >= $(deletesThreshold))
      .mapValues(derivedWords => (derivedWords._1.toList, derivedWords._2))
      .toMap
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    require(!dataset.rdd.isEmpty(), "Dataset for training is empty")

    validateDataSet(dataset)

    val possibleDict = get(dictionary).map(d => ResourceHelper.getWordCount(d))

    val trainDataSet =
      dataset.select(getInputCols.head).as[Array[Annotation]]
        .flatMap(_.map(_.result))

    val wordFrequencies =
      trainDataSet.groupBy("value").count()
        .filter(s"count(value) >= ${$(frequencyThreshold)}").as[(String, Long)].collect.toList

    val derivedWords =
      derivedWordDistances(wordFrequencies, $(maxEditDistance))

    val longestWordLength =
      trainDataSet.agg(max(length(col("value")))).head().getInt(0)

    val model =
      new SymmetricDeleteModel()
        .setDerivedWords(derivedWords)
        .setLongestWordLength(longestWordLength)

    if (possibleDict.isDefined) {
      val min = wordFrequencies.minBy(_._2)._2
      val max = wordFrequencies.maxBy(_._2)._2
      model.setMinFrequency(min)
      model.setMaxFrequency(max)
      model.setDictionary(possibleDict.get.toMap)
    }

    model
  }

  private def validateDataSet(dataset: Dataset[_]): Unit = {
    try {
      dataset.select(getInputCols.head).as[Array[Annotation]]
    }
    catch {
      case exception: AnalysisException =>
        if (exception.getMessage == "need an array field but got string;") {
          throw new IllegalArgumentException("Train dataset must have an array annotation type column")
        }
        throw exception
    }
  }

}
// This objects reads the class' properties, it enables reading the model after it is stored
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
