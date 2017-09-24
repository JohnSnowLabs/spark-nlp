package com.jsl.nlp.annotators.spell.norvig

import com.jsl.nlp.annotators.{Normalizer, RegexTokenizer}
import com.jsl.nlp.{AnnotatorApproach, DocumentAssembler, Finisher}
import com.jsl.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}


class NorvigSweetingApproach(override val uid: String)
  extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.jsl.nlp.AnnotatorType._

  override val description: String = "Spell checking algorithm inspired on Norvig model"

  val corpusPath = new Param[String](this, "corpusPath", "path to text corpus for learning")
  val corpusCol = new Param[String](this, "corpusCol", "dataset corpus text column")
  val dictPath = new Param[String](this, "dictPath", "path to dictionary of words")
  val slangPath = new Param[String](this, "slangPath", "path to custom dictionaries")

  setDefault(dictPath, "/spell/words.txt")
  setDefault(slangPath, "/spell/slangs.txt")

  setDefault(caseSensitive, false)
  setDefault(doubleVariants, false)
  setDefault(shortCircuit, false)

  override val annotatorType: AnnotatorType = SPELL

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SPELL"))

  def setDictPath(value: String): this.type = set(dictPath, value)

  def setCorpusCol(value: String): this.type = set(corpusCol, value)

  def setCorpusPath(value: String): this.type = set(corpusPath, value)

  def setSlangPath(value: String): this.type = set(slangPath, value)

  private def datasetWordCount(dataset: Dataset[_]): Map[String, Int] = {
    import dataset.sparkSession.implicits._
    val wordCount = dataset.sparkSession.sparkContext.broadcast(MMap.empty[String, Int].withDefaultValue(0))
    val documentAssembler = new DocumentAssembler()
      .setInputCol($(corpusCol))
    val tokenizer = new RegexTokenizer()
      .setInputCols("document")
      .setOutputCol("token")
    val normalizer = new Normalizer()
      .setInputCols("token")
      .setOutputCol("normal")
    val finisher = new Finisher()
      .setInputCols("normal")
      .setOutputCols("finished")
      .setAnnotationSplitSymbol("--")
    new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, normalizer, finisher))
      .fit(dataset)
      .transform(dataset)
      .select("finished").as[String]
      .foreach(text => text.split("--").foreach(t => {
        wordCount.value(t) += 1
      }))
    val result = wordCount.value.toMap
    wordCount.destroy()
    result
  }

  override def train(dataset: Dataset[_]): NorvigSweetingModel = {
    val loadWords = ResourceHelper.wordCount($(dictPath), "txt")
    val corpusWordCount =
      if (get(corpusPath).isDefined) {
        ResourceHelper.wordCount($(corpusPath), "txt")
      } else if (get(corpusCol).isDefined) {
        datasetWordCount(dataset)
      }
      else {
      Map.empty[String, Int]
      }
    val loadSlangs = ResourceHelper.parseKeyValueText($(slangPath), "txt", ",")
    new NorvigSweetingModel()
      .setWordCount(loadWords.toMap ++ corpusWordCount)
      .setCustomDict(loadSlangs)
      .setDoubleVariants($(doubleVariants))
      .setCaseSensitive($(caseSensitive))
      .setShortCircuit($(shortCircuit))
  }

}
object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]