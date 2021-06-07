package com.johnsnowlabs.nlp.annotators.spell.norvig

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{AnalysisException, Dataset}

/** This annotator retrieves tokens and makes corrections automatically if not found in an English dictionary. Inspired by Norvig model
  *
  * Inspired on [[https://github.com/wolfgarbe/SymSpell]]
  *
  * The Symmetric Delete spelling correction algorithm reduces the complexity of edit candidate generation and
  * dictionary lookup for a given Damerau-Levenshtein distance. It is six orders of magnitude faster
  * (than the standard approach with deletes + transposes + replaces + inserts) and language independent.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/spell/norvig/NorvigSweetingTestSpec.scala]] for further reference on how to use this API
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
class NorvigSweetingApproach(override val uid: String)
  extends AnnotatorApproach[NorvigSweetingModel]
    with NorvigSweetingParams {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Spell checking algorithm inspired on Norvig model */
  override val description: String = "Spell checking algorithm inspired on Norvig model"
  /** file with a list of correct words
    *
    * @group param
    **/
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

  /** External path to file with properly spelled words, tokenPattern is the regex pattern to identify them in text, readAs LINE_BY_LINE or SPARK_DATASET, with options passed to Spark reader if the latter is set. dictionary needs 'tokenPattern' regex in dictionary for separating words
    *
    * @group setParam
    **/
  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "dictionary needs 'tokenPattern' regex in dictionary for separating words")
    set(dictionary, value)
  }

  /** Path to file with properly spelled words, tokenPattern is the regex pattern to identify them in text, readAs LINE_BY_LINE or SPARK_DATASET, with options passed to Spark reader if the latter is set. dictionary needs 'tokenPattern' regex in dictionary for separating words
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
  /** Output annotator type : TOKEN
    *
    * @group anno
    **/
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