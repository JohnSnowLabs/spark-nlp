/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ReadAs, ResourceHelper, ExternalResource}
import com.johnsnowlabs.nlp.{AnnotatorType, AnnotatorApproach}
import com.johnsnowlabs.util.TrainingHelper.hasColumn
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{explode, udf, collect_set}

import scala.collection.mutable

/** Class to find lemmas out of words with the objective of returning a base dictionary word.
  * Retrieves the significant part of a word. A dictionary of predefined lemmas must be provided
  * with `setDictionary`. The dictionary can be set in either in the form of a delimited text file
  * or directly as an [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
  * Pretrained models can be loaded with [[LemmatizerModel LemmatizerModel.pretrained]].
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Lemmatization Models Hub]]. For extended examples
  * of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/italian/Train-Lemmatizer-Italian.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/LemmatizerTestSpec.scala LemmatizerTestSpec]].
  *
  * ==Example==
  * In this example, the lemma dictionary `lemmas_small.txt` has the form of
  * {{{
  * ...
  * pick	->	pick	picks	picking	picked
  * peck	->	peck	pecking	pecked	pecks
  * pickle	->	pickle	pickles	pickled	pickling
  * pepper	->	pepper	peppers	peppered	peppering
  * ...
  * }}}
  * where each key is delimited by `->` and values are delimited by `\t`
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.Tokenizer
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.Lemmatizer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentenceDetector = new SentenceDetector()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("token")
  *
  * val lemmatizer = new Lemmatizer()
  *   .setInputCols(Array("token"))
  *   .setOutputCol("lemma")
  *   .setDictionary("src/test/resources/lemma-corpus-small/lemmas_small.txt", "->", "\t")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     sentenceDetector,
  *     tokenizer,
  *     lemmatizer
  *   ))
  *
  * val data = Seq("Peter Pipers employees are picking pecks of pickled peppers.")
  *   .toDF("text")
  *
  * val result = pipeline.fit(data).transform(data)
  * result.selectExpr("lemma.result").show(false)
  * +------------------------------------------------------------------+
  * |result                                                            |
  * +------------------------------------------------------------------+
  * |[Peter, Pipers, employees, are, pick, peck, of, pickle, pepper, .]|
  * +------------------------------------------------------------------+
  * }}}
  * @see
  *   [[LemmatizerModel]] for the instantiated model and pretrained models.
  * @param uid
  *   required internal uid provided by constructor
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio anno  1
  * @groupprio param  2
  * @groupprio setParam  3
  * @groupprio getParam  4
  * @groupprio Ungrouped 5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class Lemmatizer(override val uid: String) extends AnnotatorApproach[LemmatizerModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Retrieves the significant part of a word */
  override val description: String = "Retrieves the significant part of a word"

  def this() = this(Identifiable.randomUID("LEMMATIZER"))

  /** External dictionary to be used by the lemmatizer, which needs '`keyDelimiter`' and
    * '`valueDelimiter`' for parsing the resource
    * ==Example==
    * {{{
    * ...
    * pick	->	pick	picks	picking	picked
    * peck	->	peck	pecking	pecked	pecks
    * pickle	->	pickle	pickles	pickled	pickling
    * pepper	->	pepper	peppers	peppered	peppering
    * ...
    * }}}
    * where each key is delimited by `->` and values are delimited by `\t`
    * @group param
    */
  val dictionary: ExternalResourceParam = new ExternalResourceParam(
    this,
    "dictionary",
    "External dictionary to be used by the lemmatizer, which needs 'keyDelimiter' and 'valueDelimiter' for parsing the resource")

  /** Output annotator type : TOKEN
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = TOKEN

  /** Input annotator type : TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** Column that correspends to CoNLLU(formCol=) output
    *
    * @group param
    */
  val formCol =
    new Param[String](this, "formCol", "Column that correspends to CoNLLU(formCol=) output")

  /** @group setParam */
  def setFormCol(value: String): this.type = set(formCol, value)

  /** @group getParam */
  def getFormCol: String = $(formCol)

  /** Column that correspends to CoNLLU(lemmaCol=) output
    *
    * @group param
    */
  val lemmaCol =
    new Param[String](this, "lemmaCol", "Column that correspends to CoNLLU(lemmaCol=) output")

  /** @group setParam */
  def setLemmaCol(value: String): this.type = set(lemmaCol, value)

  /** @group getParam */
  def getLemmaCol: String = $(lemmaCol)

  /** External dictionary to be used by the lemmatizer
    * @group getParam
    */
  def getDictionary: ExternalResource = $(dictionary)

  /** External dictionary already in the form of [[ExternalResource]], for which the Map member
    * `options` has entries defined for `"keyDelimiter"` and `"valueDelimiter"`.
    * ==Example==
    * {{{
    * val resource = ExternalResource(
    *   "src/test/resources/regex-matcher/rules.txt",
    *   ReadAs.TEXT,
    *   Map("keyDelimiter" -> "->", "valueDelimiter" -> "\t")
    * )
    * val lemmatizer = new Lemmatizer()
    *   .setInputCols(Array("token"))
    *   .setOutputCol("lemma")
    *   .setDictionary(resource)
    * }}}
    * @group setParam
    */
  def setDictionary(value: ExternalResource): this.type = {
    require(
      value.options.contains("keyDelimiter") && value.options.contains("valueDelimiter"),
      "Lemmatizer dictionary requires options with 'keyDelimiter' and 'valueDelimiter'")
    set(dictionary, value)
  }

  /** External dictionary to be used by the lemmatizer, which needs `keyDelimiter` and
    * `valueDelimiter` for parsing the resource
    * @group setParam
    */
  def setDictionary(
      path: String,
      keyDelimiter: String,
      valueDelimiter: String,
      readAs: ReadAs.Format = ReadAs.TEXT,
      options: Map[String, String] = Map("format" -> "text")): this.type =
    set(
      dictionary,
      ExternalResource(
        path,
        readAs,
        options ++ Map("keyDelimiter" -> keyDelimiter, "valueDelimiter" -> valueDelimiter)))

  setDefault(
    dictionary -> ExternalResource("", ReadAs.TEXT, Map()),
    formCol -> "form",
    lemmaCol -> "lemma")

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): LemmatizerModel = {
    if (getDictionary.path != "") {
      new LemmatizerModel()
        .setLemmaDict(ResourceHelper.flattenRevertValuesAsKeys($(dictionary)))
    } else {
      validateColumn(dataset, getFormCol, AnnotatorType.TOKEN)
      validateColumn(dataset, getLemmaCol, AnnotatorType.TOKEN)
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

    val formColumn = getFormCol + ".result"
    val lemmaColumn = getLemmaCol + ".result"

    val lemmaDataSet = dataset
      .select(dataset.col(formColumn).as("forms"), dataset.col(lemmaColumn).as("lemmas"))
      .withColumn("forms_lemmas", explode(arraysZip($"forms", $"lemmas")))
      .withColumn("token_training", $"forms_lemmas._1")
      .withColumn("lemma_training", $"forms_lemmas._2")
      .groupBy("lemma_training")
      .agg(collect_set("token_training").as("tokens"))

    val dictionary = lemmaDataSet
      .select("lemma_training", "tokens")
      .rdd
      .flatMap { row =>
        val lemma: String = row.get(0).asInstanceOf[String]
        val tokens: Seq[String] = row.get(1).asInstanceOf[mutable.WrappedArray[String]]
        tokens.flatMap(t => Map(t -> lemma))
      }
      .collect()
      .toMap
    dictionary
  }

  def arraysZip: UserDefinedFunction = udf { (forms: Seq[String], lemmas: Seq[String]) =>
    forms.zip(lemmas)
  }

}

/** This is the companion object of [[Lemmatizer]]. Please refer to that class for the
  * documentation.
  */
object Lemmatizer extends DefaultParamsReadable[Lemmatizer]
