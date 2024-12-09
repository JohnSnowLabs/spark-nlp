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

import com.johnsnowlabs.ml.ai.{MPNetClassification, MergeTokenStrategy}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{loadTextAsset, modelSanityCheck, notSupportedEngineError}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** MPNetForQuestionAnswering can load MPNet Models with a span classification head on top for
  * extractive question-answering tasks like SQuAD (a linear layer on top of the hidden-states
  * output to compute span start logits and span end logits).
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val spanClassifier = MPNetForQuestionAnswering.pretrained()
  *   .setInputCols(Array("document_question", "document_context"))
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"mpnet_base_question_answering_squad2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForQuestionAnsweringTestSpec.scala MPNetForQuestionAnsweringTestSpec]].
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
  * val questionAnswering = MPNetForQuestionAnswering.pretrained()
  *   .setInputCols(Array("document_question", "document_context"))
  *   .setOutputCol("answer")
  *   .setCaseSensitive(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   document,
  *   questionAnswering
  * ))
  *
  * val data = Seq("What's my name?", "My name is Clara and I live in Berkeley.").toDF("question", "context")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("label.result").show(false)
  * +---------------------+
  * |result               |
  * +---------------------+
  * |[Clara]              |
  * ++--------------------+
  * }}}
  *
  * @see
  *   [[MPNetForSequenceClassification]] for sequence-level classification
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
class MPNetForQuestionAnswering(override val uid: String)
    extends AnnotatorModel[MPNetForQuestionAnswering]
    with HasBatchedAnnotate[MPNetForQuestionAnswering]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("MPNetForQuestionAnswering"))

  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)

  /** Output Annotator Types: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CHUNK

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  def padTokenId: Int = {
    $$(vocabulary)("<pad>")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Max sentence length to process (Default: `384`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "MPNet models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[MPNetClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): MPNetForQuestionAnswering = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MPNetClassification(
            tensorflowWrapper = None,
            onnxWrapper = onnxWrapper,
            openvinoWrapper = openvinoWrapper,
            sentenceStartTokenId = sentenceStartTokenId,
            sentenceEndTokenId = sentenceEndTokenId,
            tags = Map.empty[String, Int],
            signatures = getSignatures,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: MPNetClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `true`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  setDefault(batchSize -> 8, maxSentenceLength -> 384, caseSensitive -> false)

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map(annotations => {
      val documents = annotations
        .filter(_.annotatorType == AnnotatorType.DOCUMENT)
        .toSeq

      if (documents.nonEmpty) {
        getModelIfNotSet.predictSpan(
          documents,
          $(maxSentenceLength),
          $(caseSensitive),
          MergeTokenStrategy.vocab)
      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_MPNet_classification"

    getEngine match {
      case TensorFlow.name =>
        throw new NotImplementedError("Tensorflow models are not supported.")
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          MPNetForQuestionAnswering.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          MPNetForQuestionAnswering.openvinoFile)
    }
  }
}

trait ReadablePretrainedMPNetForQAModel
    extends ParamsAndFeaturesReadable[MPNetForQuestionAnswering]
    with HasPretrained[MPNetForQuestionAnswering] {
  override val defaultModelName: Some[String] = Some("mpnet_base_question_answering_squad2")

  /** Java compliant-overrides */
  override def pretrained(): MPNetForQuestionAnswering = super.pretrained()

  override def pretrained(name: String): MPNetForQuestionAnswering = super.pretrained(name)

  override def pretrained(name: String, lang: String): MPNetForQuestionAnswering =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): MPNetForQuestionAnswering =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMPNetForQuestionAnsweringDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[MPNetForQuestionAnswering] =>
  override val onnxFile: String = "mpnet_question_answering_onnx"
  override val openvinoFile: String = "mpnet_question_answering_openvino"

  def readModel(instance: MPNetForQuestionAnswering, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "mpnet_qa_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, Some(onnxWrapper), None)


      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "distilbert_qa_classification_openvino")
        instance.setModelIfNotSet(spark, None, Some(openvinoWrapper))

      case _ =>
        throw new NotImplementedError("Tensorflow models are not supported.")
    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MPNetForQuestionAnswering = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new MPNetForQuestionAnswering()
      .setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        throw new NotImplementedError("Tensorflow models are not supported.")
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrapper), None)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(ovWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MPNetForQuestionAnswering]]. Please refer to that class for
  * the documentation.
  */
object MPNetForQuestionAnswering
    extends ReadablePretrainedMPNetForQAModel
    with ReadMPNetForQuestionAnsweringDLModel
