/*
 * Copyright 2017-2024 John Snow Labs
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

import com.johnsnowlabs.ml.ai.BertClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** BertForMultipleChoice can load BERT Models with a multiple choice classification head on top
 *  (a linear layer on top of the pooled output and a softmax) e.g. for RocStories/SWAG tasks.
 *
 * Pretrained models can be loaded with `pretrained` of the companion object:
 * {{{
 * val spanClassifier = BertForMultipleChoice.pretrained()
 *   .setInputCols(Array("document_question", "document_context"))
 *   .setOutputCol("answer")
 * }}}
 * The default model is `"bert_base_uncased_multiple_choice"`, if no name is provided.
 *
 * For available pretrained models please see the
 * [[https://sparknlp.org/models?task=Multiple+Choice Models Hub]].
 *
 * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
 * see which models are compatible and how to import them see
 * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
 * examples, see
 * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/BertForMultipleChoiceTestSpec.scala BertForMultipleChoiceTestSpec]].
 *
 * ==Example==
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.base._
 * import com.johnsnowlabs.nlp.annotator._
 * import org.apache.spark.ml.Pipeline
 *
 * val document = new MultiDocumentAssembler()
 *   .setInputCols("question", "context")
 *   .setOutputCols("document_question", "document_context")
 *
 * val questionAnswering = BertForMultipleChoice.pretrained()
 *   .setInputCols(Array("document_question", "document_context"))
 *   .setOutputCol("answer")
 *   .setCaseSensitive(false)
 *
 * val pipeline = new Pipeline().setStages(Array(
 *   document,
 *   questionAnswering
 * ))
 *
 * val data = Seq("The Eiffel Tower is located in which country?", "Germany, France, Italy").toDF("question", "context")
 * val result = pipeline.fit(data).transform(data)
 *
 * result.select("answer.result").show(false)
 * +---------------------+
 * |result               |
 * +---------------------+
 * |[France]              |
 * ++--------------------+
 * }}}
 *
 * @see
 *   [[BertForQuestionAnswering]] for Question Answering tasks
 * @see
 *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
 *   based classifiers
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

class BertForMultipleChoice(override val uid: String)
    extends AnnotatorModel[BertForMultipleChoice]
    with HasBatchedAnnotate[BertForMultipleChoice]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("BertForMultipleChoice"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CHUNK

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Max sentence length to process (Default: `512`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "BERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  val choicesDelimiter =
    new Param[String](this, "choicesDelimiter", "Delimiter character use to split the choices")

  def setChoicesDelimiter(value: String): this.type = set(choicesDelimiter, value)

  private var _model: Option[Broadcast[BertClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): BertForMultipleChoice = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new BertClassification(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = None,
            tags = Map.empty[String, Int],
            signatures = None,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: BertClassification = _model.get.value

  setDefault(
    batchSize -> 4,
    maxSentenceLength -> 512,
    caseSensitive -> false,
    choicesDelimiter -> ",")

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations in batches that correspond to inputAnnotationCols generated by previous
    *   annotators if any
    * @return
    *   any number of annotations processed for every batch of input annotations. Not necessary
    *   one to one relationship
    *
    * IMPORTANT: !MUST! return sequences of equal lengths !! IMPORTANT: !MUST! return sentences
    * that belong to the same original row !! (challenging)
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map(annotations => {
      if (annotations.nonEmpty) {
        getModelIfNotSet.predictSpanMultipleChoice(
          annotations,
          $(choicesDelimiter),
          $(maxSentenceLength),
          $(caseSensitive))
      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_bert_multiple_choice_classification",
          BertForMultipleChoice.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          BertForMultipleChoice.openvinoFile)

    }
  }

}

trait ReadablePretrainedBertForMultipleChoiceModel
    extends ParamsAndFeaturesReadable[BertForMultipleChoice]
    with HasPretrained[BertForMultipleChoice] {
  override val defaultModelName: Some[String] = Some("bert_base_uncased_multiple_choice")

  /** Java compliant-overrides */
  override def pretrained(): BertForMultipleChoice = super.pretrained()

  override def pretrained(name: String): BertForMultipleChoice = super.pretrained(name)

  override def pretrained(name: String, lang: String): BertForMultipleChoice =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BertForMultipleChoice =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBertForMultipleChoiceModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[BertForMultipleChoice] =>

  override val onnxFile: String = "bert_mc_classification_onnx"
  override val openvinoFile: String = "bert_mc_classification_openvino"

  def readModel(instance: BertForMultipleChoice, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "bert_mc_classification_onnx")
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)
      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "bert_mc_classification_ov")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BertForMultipleChoice = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    val annotatorModel = new BertForMultipleChoice().setVocabulary(vocabs)
    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None)
      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

/** This is the companion object of [[BertForMultipleChoice]]. Please refer to that class for the
  * documentation.
  */
object BertForMultipleChoice
    extends ReadablePretrainedBertForMultipleChoiceModel
    with ReadBertForMultipleChoiceModel
