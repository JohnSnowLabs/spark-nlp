package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Class to find standarized lemmas from words. Uses a user-provided or default dictionary.
  *
  * Retrieves lemmas out of words with the objective of returning a base dictionary word. Retrieves the significant part of a word.
  *
  * lemmaDict: A dictionary of predefined lemmas must be provided
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/LemmatizerTestSpec.scala]] for examples of how to use this API
  *
  * @param uid required internal uid provided by constructor
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class Lemmatizer(override val uid: String) extends AnnotatorApproach[LemmatizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Retrieves the significant part of a word  */
  override val description: String = "Retrieves the significant part of a word"
  /** lemmatizer external dictionary, needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text
    *
    * @group param
    **/
  val dictionary: ExternalResourceParam = new ExternalResourceParam(this, "dictionary", "lemmatizer external dictionary, needs 'keyDelimiter' and 'valueDelimiter' in options for parsing target text")

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

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  /** Path and options to lemma dictionary, in lemma vs possible words format. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET. lemmatizer external dictionary
    *
    * @group getParam
    **/
  def getDictionary: ExternalResource = $(dictionary)

  /** setDictionary(path, keyDelimiter, valueDelimiter, readAs, options): Path and options to lemma dictionary, in lemma vs possible words format. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.  lemmatizer external dictionary
    *
    * @group setParam
    **/
  def setDictionary(value: ExternalResource): this.type = {
    require(value.options.contains("keyDelimiter") && value.options.contains("valueDelimiter"),
      "Lemmatizer dictionary requires options with 'keyDelimiter' and 'valueDelimiter'")
    set(dictionary, value)
  }

  /** setDictionary(path, keyDelimiter, valueDelimiter, readAs, options): Path and options to lemma dictionary, in lemma vs possible words format. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.  lemmatizer external dictionary
    *
    * @group setParam
    **/
  def setDictionary(
                     path: String,
                     keyDelimiter: String,
                     valueDelimiter: String,
                     readAs: ReadAs.Format = ReadAs.TEXT,
                     options: Map[String, String] = Map("format" -> "text")): this.type =
    set(dictionary, ExternalResource(path, readAs, options ++ Map("keyDelimiter" -> keyDelimiter, "valueDelimiter" -> valueDelimiter)))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): LemmatizerModel = {
    new LemmatizerModel()
      .setLemmaDict(ResourceHelper.flattenRevertValuesAsKeys($(dictionary)))
  }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
