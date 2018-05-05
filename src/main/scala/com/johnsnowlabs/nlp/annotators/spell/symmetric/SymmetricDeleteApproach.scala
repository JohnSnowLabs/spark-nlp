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
  def updateDerivedWords(derivedWords: MMap[String, (ListBuffer[String], Long)],
                         word: String, maxEditDistance: Int, longestWordLength: Int
                         ): Int = {

    var newLongestWordLength = longestWordLength

    if (derivedWords(word)._2 == 0) {
      derivedWords(word) = (ListBuffer[String](), 1)
      newLongestWordLength = word.length.max(longestWordLength)
    } else{
      var count: Long = derivedWords(word)._2
      // increment count of word in corpus
      count += 1
      derivedWords(word) = (derivedWords(word)._1, count)
    }

    if (derivedWords(word)._2 == 1){
      val deletes = getDeletes(word, maxEditDistance)

      deletes.foreach( item => {
        if (derivedWords.contains(item)){
          // add (correct) word to delete's suggested correction list
          derivedWords(item)._1 += word
        } else {
          // note frequency of word in corpus is not incremented
          val wordFrequency = new ListBuffer[String]
          wordFrequency += word
          derivedWords(item) = (wordFrequency, 0)
        }
      }) // End deletes.foreach
    }
    newLongestWordLength
  }

  /** Created by danilo 26/04/2018
    * Computes derived words from a word
    * */
  def derivedWordDistances(externalResource: List[String], maxEditDistance: Int
                          ): WordFeatures = {

    val regex = $(corpus).options("tokenPattern").r
    val derivedWords = MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0)
    val wordFeatures = WordFeatures(derivedWords, 0)
    var longestWordLength = wordFeatures.longestWordLength

    externalResource.foreach(line => {
      val tokenizeWords = regex.findAllMatchIn(line).map(_.matched).toList
      tokenizeWords.foreach(word => {
        longestWordLength = updateDerivedWords(wordFeatures.derivedWords, word, maxEditDistance, longestWordLength)
      })
    })
    wordFeatures.longestWordLength = longestWordLength
    wordFeatures
  }

  case class WordFeatures(var derivedWords: MMap[String, (ListBuffer[String], Long)],
                          var longestWordLength: Int)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SymmetricDeleteModel = {

    var externalResource = List[String]()
    val possibleDict = get(dictionary).map(d => ResourceHelper.wordCount(d))
    var wordFeatures = WordFeatures(MMap.empty, 0)

    if (get(corpus).isDefined) {
      externalResource = ResourceHelper.parseLines($(corpus)).map(_.toLowerCase).toList
      printf("Number of words with training file: %d\n", externalResource.size)
      wordFeatures = derivedWordDistances(externalResource = externalResource, maxEditDistance = $(maxEditDistance))
      printf("Derive words size with training file %d\n", wordFeatures.derivedWords.size)
    } else {
      println("Training from dataset under construction")
      import ResourceHelper.spark.implicits._
      import org.apache.spark.sql.functions._
      dataset.show(5, false)
      dataset.select($(inputCols).head).show(5, false)
      dataset.select($(inputCols).head).as[Array[Annotation]]
        .flatMap(_.map(_.result)).show(20, false)

      val trainDataset = dataset.select($(inputCols).head).as[Array[Annotation]]
                        .flatMap(_.map(_.result))
      trainDataset.show()
      /*println("Size without grouping")
      println(trainDataset.count())
      println("Size grouping")*/
      val resourceList = trainDataset.collect.toList
      printf("Number of words with training dataset: %d\n", resourceList.size)
      wordFeatures = getDerivedWordsv2(resourceList, $(maxEditDistance))
      printf("Derive words size with training dataset: %d\n", wordFeatures.derivedWords.size)
      /****************************************** My version ********************************/
      /*
      val resourceList = trainDataset.groupBy("value").count().as[(String, Long)].collect.toList
      val df = trainDataset.withColumn("length", length(col("value")))
      val longestWordLength = df.agg(max(col("length"))).head().getInt(0)
      //println(longestWordLength)
      wordFeatures = getDerivedWords(resourceList, $(maxEditDistance))
      wordFeatures.longestWordLength = longestWordLength*/



      //println(wordFeatures.derivedWords.size)
      // https://www.balabit.com/blog/spark-scala-dataset-tutorial
      // https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-udfs.html
      // https://medium.com/@mrpowers/spark-user-defined-functions-udfs-6c849e39443b
      /*var trainDatasetNew =  trainDataset.withColumn("length", length(col("value")))
      trainDatasetNew.show(5)
      //sw_ds.withColumn("Jedi_Species", createTuple2[String, String].apply(sw_ds("jedi"), sw_ds("species")))
      trainDatasetNew = trainDatasetNew.withColumn("tuple",
                                                   createTuple2[String, String].apply(col("value"), col("length")))
      trainDatasetNew.show(5, false)
      trainDatasetNew = trainDatasetNew.withColumn("upper", upperUDF(col("value")))
      trainDatasetNew.show(5, false)
      println("End training from dataset")*/
    }


    val model = new SymmetricDeleteModel()
      .setDerivedWords(wordFeatures.derivedWords.toMap)
      .setLongestWordLength(wordFeatures.longestWordLength)

    if (possibleDict.isDefined)
      model.setDictionary(possibleDict.get.toMap)

    model
  }

  /**********************  mdba Temp just for testing ******************************/

  import scala.reflect.runtime.universe.TypeTag
  import org.apache.spark.sql.functions.udf
  def createTuple2[Type_x: TypeTag, Type_y: TypeTag] =
    udf[(Type_x, Type_y), Type_x, Type_y]((x: Type_x, y: Type_y) => (x, y))

  // Define a regular Scala function
  def upperFunction: String => String = _.toUpperCase
  // Define a UDF that wraps the upper Scala function defined above
  // You could also define the function in place, i.e. inside udf
  // but separating Scala functions from Spark SQL's UDFs allows for easier testing
  import org.apache.spark.sql.functions.udf
  def upperUDF = udf(upperFunction)

  def getDerivedWords(words: List[(String, Long)], maxEditDistance: Int): WordFeatures = {

    //val derivedWords = MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0)
    val derivedWords = collection.mutable.Map() ++ words.map{a => (a._1, (ListBuffer[String](),a._2))}.toMap
    val wordFeatures = WordFeatures(derivedWords, 0)
    //printf("Derive words before loop %d\n", derivedWords.size)
    words.foreach(word =>{
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
    //printf("Derive words after loop %d\n", derivedWords.size)
    wordFeatures.derivedWords = derivedWords
    wordFeatures
  }

  def getDerivedWordsv2(words: List[String], maxEditDistance: Int): WordFeatures = {

    //val derivedWords = MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0)
    val derivedWords = MMap.empty[String, (ListBuffer[String], Long)].withDefaultValue(ListBuffer[String](), 0)
    val wordFeatures = WordFeatures(derivedWords, 0)
    var longestWordLength = wordFeatures.longestWordLength

    words.foreach(word =>{
      longestWordLength = updateDerivedWords(wordFeatures.derivedWords, word, maxEditDistance, longestWordLength)
    })
    wordFeatures.longestWordLength = longestWordLength
    wordFeatures
  }

  /**********************  mdba Temp just for testing ******************************/

}
// This objects reads the class' properties, it enables reding the model after it is stored
object SymmetricDeleteApproach extends DefaultParamsReadable[SymmetricDeleteApproach]
