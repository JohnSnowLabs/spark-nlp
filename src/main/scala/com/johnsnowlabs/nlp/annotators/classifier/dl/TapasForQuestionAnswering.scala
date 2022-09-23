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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowTapas, TensorflowWrapper}
import com.johnsnowlabs.nlp.base.TableAssembler
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, HasPretrained, ParamsAndFeaturesReadable}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import java.io.File

/** TapasForQuestionAnswering is an implementation of TaPas - a BERT-based model specifically
  * designed for answering questions about tabular data. It takes TABLE and DOCUMENT annotations
  * as input and tries to answer the questions in the document by using the data from the table.
  * The model is based in BertForQuestionAnswering and shares all its parameters with it.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val tapas = TapasForQuestionAnswering.pretrained()
  *   .setInputCols(Array("document_question", "table"))
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"table_qa_tapas_base_finetuned_wtq"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Table+Question+Understanding Models Hub]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  *  val questions =
  *    """
  *     |Who earns 100,000,000?
  *     |Who has more money?
  *     |How old are they?
  *     |""".stripMargin.trim
  *
  *  val jsonData =
  *    """
  *     |{
  *     | "header": ["name", "money", "age"],
  *     | "rows": [
  *     |   ["Donald Trump", "$100,000,000", "75"],
  *     |   ["Elon Musk", "$20,000,000,000,000", "55"]
  *     | ]
  *     |}
  *     |""".stripMargin.trim
  *
  *  val data = Seq((jsonData, questions))
  *   .toDF("json_table", "questions")
  *   .repartition(1)
  *
  * val docAssembler = new MultiDocumentAssembler()
  *   .setInputCols("json_table", "questions")
  *   .setOutputCols("document_table", "document_questions")
  *
  * val sentenceDetector = SentenceDetectorDLModel
  *   .pretrained()
  *   .setInputCols(Array("document_questions"))
  *   .setOutputCol("question")
  *
  * val tableAssembler = new TableAssembler()
  *   .setInputFormat("json")
  *   .setInputCols(Array("document_table"))
  *   .setOutputCol("table")
  *
  * val tapas = TapasForQuestionAnswering
  *   .pretrained()
  *   .setInputCols(Array("question", "table"))
  *   .setOutputCol("answer")
  *
  * val pipeline = new Pipeline()
  *   .setStages(
  *     Array(
  *       docAssembler,
  *       sentenceDetector,
  *       tableAssembler,
  *        tapas))
  *
  * val pipelineModel = pipeline.fit(data)
  * val result = pipeline.fit(data).transform(data)
  *
  * result
  *   .selectExpr("explode(answer) as answer")
  *   .selectExpr(
  *     "answer.metadata.question",
  *     "answer.result")
  *
  * +-----------------------+----------------------------------------+
  * |question               |result                                  |
  * +-----------------------+----------------------------------------+
  * |Who earns 100,000,000? |Donald Trump                            |
  * |Who has more money?    |Elon Musk                               |
  * |How much they all earn?|COUNT($100,000,000, $20,000,000,000,000)|
  * |How old are they?      |AVERAGE(75, 55)                         |
  * +-----------------------+----------------------------------------+
  * }}}
  *
  * @see
  *   [[https://aclanthology.org/2020.acl-main.398/]] for more details about the TaPas model
  * @see
  *   [[TableAssembler]] for loading tabular data
  * @see
  *   [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of
  *   transformer based classifiers
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */

class TapasForQuestionAnswering(override val uid: String) extends BertForQuestionAnswering(uid) {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("TapasForQuestionAnswering"))

  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.TABLE, AnnotatorType.DOCUMENT)

  private var _model: Option[Broadcast[TensorflowTapas]] = None

  /** @group setParam */
  override def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper): TapasForQuestionAnswering = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowTapas(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            tags = Map.empty[String, Int],
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  override def getModelIfNotSet: TensorflowTapas = _model.get.value

  setDefault(batchSize -> 2, maxSentenceLength -> 512, caseSensitive -> false)

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    batchedAnnotations.map(annotations => {

      val questions = annotations
        .filter(_.annotatorType == AnnotatorType.DOCUMENT)
        .toSeq

      val tables = annotations
        .filter(_.annotatorType == AnnotatorType.TABLE)
        .toSeq

      if (questions.nonEmpty) {
        tables.flatMap { table =>
          {
            getModelIfNotSet.predictTapasSpan(
              questions,
              table,
              $(maxSentenceLength),
              $(caseSensitive),
              0.5f)
          }
        }
      } else {
        Seq.empty[Annotation]
      }
    })
  }

}

trait ReadablePretrainedTapasForQAModel
    extends ParamsAndFeaturesReadable[TapasForQuestionAnswering]
    with HasPretrained[TapasForQuestionAnswering] {
  override val defaultModelName: Some[String] = Some("table_qa_tapas_base_finetuned_wtq")

  /** Java compliant-overrides */
  override def pretrained(): TapasForQuestionAnswering = super.pretrained()

  override def pretrained(name: String): TapasForQuestionAnswering = super.pretrained(name)

  override def pretrained(name: String, lang: String): TapasForQuestionAnswering =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): TapasForQuestionAnswering = super.pretrained(name, lang, remoteLoc)
}

trait ReadTapasForQATensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[TapasForQuestionAnswering] =>

  override val tfFile: String = "bert_classification_tensorflow"

  def readTensorflow(
      instance: TapasForQuestionAnswering,
      path: String,
      spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_bert_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): TapasForQuestionAnswering = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath")

    val vocabPath = new File(tfModelPath + "/assets", "vocab.txt")
    require(
      vocabPath.exists(),
      s"Vocabulary file vocab.txt not found in folder $tfModelPath/assets/")

    val vocabResource =
      new ExternalResource(vocabPath.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val (wrapper, signatures) =
      TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new TapasForQuestionAnswering()
      .setVocabulary(words)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}

/** This is the companion object of [[BertForQuestionAnswering]]. Please refer to that class for
  * the documentation.
  */
object TapasForQuestionAnswering
    extends ReadablePretrainedTapasForQAModel
    with ReadTapasForQATensorflowModel
