package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{AnalysisException, Dataset}
import ResourceHelper.spark.implicits._

class NorvigSweetingApproach(override val uid: String)
  extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Spell checking algorithm inspired on Norvig model"

  val dictionary = new ExternalResourceParam(this, "dictionary", "file with a list of correct words")

  setDefault(
    caseSensitive -> true,
    doubleVariants -> false,
    shortCircuit -> false,
    frequencyPriority -> true,
    wordSizeIgnore -> 3,
    dupsLimit -> 2,
    reductLimit -> 3,
    intersections -> 10,
    vowelSwapLimit -> 6
  )

  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  def setDictionary(path: String,
                    tokenPattern: String = "\\S+",
                    readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                    options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("SPELL"))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): NorvigSweetingModel = {

    validateDataSet(dataset)
    val loadWords = ResourceHelper.getWordCount($(dictionary)).toMap
    val corpusWordCount: Map[String, Long] = {

        dataset.select(getInputCols.head).as[Array[Annotation]]
          .flatMap(_.map(_.result))
          .groupBy("value").count
          .as[(String, Long)]
          .collect.toMap
      }

    new NorvigSweetingModel()
      .setWordSizeIgnore($(wordSizeIgnore))
      .setDupsLimit($(dupsLimit))
      .setReductLimit($(reductLimit))
      .setIntersections($(intersections))
      .setVowelSwapLimit($(vowelSwapLimit))
      .setWordCount(loadWords ++ corpusWordCount)
      .setDoubleVariants($(doubleVariants))
      .setCaseSensitive($(caseSensitive))
      .setShortCircuit($(shortCircuit))
      .setFrequencyPriority($(frequencyPriority))
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
object NorvigSweetingApproach extends DefaultParamsReadable[NorvigSweetingApproach]