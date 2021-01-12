package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.util.TrainingHelper.hasColumn
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{collect_set, explode, udf}

import scala.collection.mutable

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

  setDefault(dictionary, ExternalResource("", ReadAs.TEXT, Map()))

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): LemmatizerModel = {
    if (getDictionary.path != "") {
      new LemmatizerModel()
        .setLemmaDict(ResourceHelper.flattenRevertValuesAsKeys($(dictionary)))
    } else {
      validateColumn(dataset, "form", AnnotatorType.TOKEN)
      validateColumn(dataset, "lemma", AnnotatorType.TOKEN)
      val dictionary = computeDictionaryFromCoNLLUDataSet(dataset)
      new LemmatizerModel()
        .setLemmaDict(dictionary)
    }
  }

  private def validateColumn(dataset: Dataset[_], column: String, annotatorType: String): Unit = {
    val message = "column required. Verify that training dataset was loaded with CoNLLU component"
    if (!hasColumn(dataset, column)) {
      throw new IllegalArgumentException(s"$column $message")
    } else {
      val datasetSchemaFields = dataset.schema.fields.find(field =>
        field.name.contains(column) && field.metadata.contains("annotatorType")
          && field.metadata.getString("annotatorType") == annotatorType)

      if (datasetSchemaFields.isEmpty) {
        throw new IllegalArgumentException(s"$column is not a $annotatorType annotator type")
      }
    }
  }

  private def computeDictionaryFromCoNLLUDataSet(dataset: Dataset[_]): Map[String, String] = {

    import dataset.sparkSession.implicits._

    val lemmaDataSet = dataset.select($"form.result".as("forms"), $"lemma.result".as("lemmas"))
      .withColumn("forms_lemmas", explode(arraysZip($"forms", $"lemmas")))
      .withColumn("token", $"forms_lemmas._1")
      .withColumn("lemma", $"forms_lemmas._2")
      .groupBy("lemma")
      .agg(collect_set("token").as("tokens"))

    val dictionary = lemmaDataSet.select("lemma", "tokens").rdd.flatMap{ row =>
      val lemma: String = row.get(0).asInstanceOf[String]
      val tokens: Seq[String] = row.get(1).asInstanceOf[mutable.WrappedArray[String]]
      tokens.flatMap( t => Map(t -> lemma))
    }.collect().toMap
    dictionary
  }

  def arraysZip: UserDefinedFunction = udf { (forms: Seq[String], lemmas: Seq[String]) => forms.zip(lemmas) }

}

object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
