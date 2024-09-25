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

import com.johnsnowlabs.ml.ai.CamemBertClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.annotators.common.{SentenceSplit, TokenizedWithSentence}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  AnnotatorType,
  HasBatchedAnnotate,
  HasCandidateLabelsProperties,
  HasCaseSensitiveProperties,
  HasClassifierActivationProperties,
  HasEngine,
  HasPretrained,
  ParamsAndFeaturesReadable
}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

class CamemBertForZeroShotClassification(override val uid: String)
    extends AnnotatorModel[CamemBertForZeroShotClassification]
    with HasBatchedAnnotate[CamemBertForZeroShotClassification]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
    with WriteSentencePieceModel
    with HasCaseSensitiveProperties
    with HasClassifierActivationProperties
    with HasEngine
    with HasCandidateLabelsProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */

  def this() = this(Identifiable.randomUID("CamemBertForZeroShotClassification"))

  /** Input Annotator Types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Output Annotator Types: CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  /** Labels used to decode predicted IDs back to string tags
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels").setProtected()

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** Returns labels used to train this model */
  def getClasses: Array[String] = {
    $$(labels).keys.toArray
  }

  /** Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document
    * by averaging probabilities in all sentences (Default: `false`).
    *
    * Due to max sequence length limit in almost all transformer models such as DeBerta (512
    * tokens), this parameter helps feeding all the sentences into the model and averaging all the
    * probabilities for the entire document instead of probabilities per sentence.
    *
    * @group param
    */
  val coalesceSentences = new BooleanParam(
    this,
    "coalesceSentences",
    "If sets to true the output of  all sentences will be averaged to one output instead of one output per sentence. Defaults to false.")

  /** @group setParam */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** @group getParam */
  def getCoalesceSentences: Boolean = $(coalesceSentences)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): CamemBertForZeroShotClassification.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "DeBerta models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[CamemBertClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper],
      spp: SentencePieceWrapper): CamemBertForZeroShotClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new CamemBertClassification(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            spp,
            configProtoBytes = None,
            tags = $$(labels),
            signatures = getSignatures,
            threshold = $(threshold))))
    }

    this
  }

  /** Whether to lowercase tokens or not (Default: `true`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    set(this.caseSensitive, value)
  }

  /** @group getParam */
  def getModelIfNotSet: CamemBertClassification = _model.get.value

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true,
    coalesceSentences -> false)

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
      val sentences = SentenceSplit.unpack(annotations).toArray
      val tokenizedSentences = TokenizedWithSentence.unpack(annotations).toArray

      if (tokenizedSentences.nonEmpty) {
        getModelIfNotSet.predictSequenceWithZeroShot(
          tokenizedSentences,
          sentences,
          $(candidateLabels),
          $(entailmentIdParam),
          $(contradictionIdParam),
          $(batchSize),
          $(maxSentenceLength),
          $(caseSensitive),
          $(coalesceSentences),
          $$(labels),
          getActivation)

      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_camembert_classification"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          CamemBertForSequenceClassification.tfFile)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          CamemBertForSequenceClassification.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          CamemBertForSequenceClassification.openvinoFile)

    }

    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      "_camembert",
      CamemBertForSequenceClassification.sppFile)

  }
}

trait ReadPretrainedCamemBertForZeroShotClassification
    extends ParamsAndFeaturesReadable[CamemBertForZeroShotClassification]
    with HasPretrained[CamemBertForZeroShotClassification] {
  override val defaultModelName: Some[String] = Some("camembert_zero_shot_classifier_xnli_onnx")
  override val defaultLang: String = "fr"

  override def pretrained(): CamemBertForZeroShotClassification = super.pretrained()

  override def pretrained(name: String): CamemBertForZeroShotClassification =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): CamemBertForZeroShotClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): CamemBertForZeroShotClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCamemBertForZeroShotClassification
    extends ReadTensorflowModel
    with ReadOnnxModel
    with ReadSentencePieceModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[CamemBertForZeroShotClassification] =>

  override val tfFile: String = "camembert_classification_tensorflow"
  override val onnxFile: String = "camembert_classification_onnx"
  override val openvinoFile: String = "camembert_classification_openvino"
  override val sppFile: String = "camembert_spp"

  def readModel(
      instance: CamemBertForZeroShotClassification,
      path: String,
      spark: SparkSession): Unit = {

    val spp = readSentencePieceModel(path, spark, "_camembert_spp", sppFile)

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_camembert_classification_tf")
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None, spp)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(
            path,
            spark,
            "camembert_zero_classification_onnx",
            zipped = true,
            useBundle = false,
            None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None, spp)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_camembert_classification_ov")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper), spp)

      case _ =>
        throw new Exception(notSupportedEngineError)

    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession): CamemBertForZeroShotClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")
    val labels = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap

    val entailmentIds = labels.filter(x => x._1.toLowerCase().startsWith("entail")).values.toArray
    val contradictionIds =
      labels.filter(x => x._1.toLowerCase().startsWith("contradict")).values.toArray

    require(
      entailmentIds.length == 1 && contradictionIds.length == 1,
      s"""This annotator supports classifiers trained on NLI datasets. You must have only at least 2 or maximum 3 labels in your dataset:

          example with 3 labels: 'contradict', 'neutral', 'entailment'
          example with 2 labels: 'contradict', 'entailment'

          You can modify assets/labels.txt file to match the above format.

          Current labels: ${labels.keys.mkString(", ")}
          """)

    val annotatorModel = new CamemBertForZeroShotClassification()
      .setLabels(labels)
      .setCandidateLabels(labels.keys.toArray)

    /* set the entailment id */
    annotatorModel.set(annotatorModel.entailmentIdParam, entailmentIds.head)
    /* set the contradiction id */
    annotatorModel.set(annotatorModel.contradictionIdParam, contradictionIds.head)
    /* set the engine */
    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(wrapper), None, None, spModel)
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel.setModelIfNotSet(spark, None, Some(onnxWrapper), None, spModel)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper), spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

/** This is the companion object of [[CamemBertForZeroShotClassification]]. Please refer to that
  * class for the documentation.
  */

object CamemBertForZeroShotClassification
    extends ReadPretrainedCamemBertForZeroShotClassification
    with ReadCamemBertForZeroShotClassification
