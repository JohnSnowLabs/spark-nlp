/*
 * Copyright 2017-2025 John Snow Labs
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

import com.johnsnowlabs.ml.ai.RoBertaClassification
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

class RoBertaForMultipleChoice(override val uid: String)
    extends AnnotatorModel[RoBertaForMultipleChoice]
    with HasBatchedAnnotate[RoBertaForMultipleChoice]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("RoBertaForMultipleChoice"))

  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] =
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

  /** Holding merges.txt coming from RoBERTa model
    *
    * @group param
    */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

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
      "RoBERTa models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  private var _model: Option[Broadcast[RoBertaClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): RoBertaForMultipleChoice = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new RoBertaClassification(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            tags = Map.empty[String, Int],
            merges = $$(merges),
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: RoBertaClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `true`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  val choicesDelimiter =
    new Param[String](this, "choicesDelimiter", "Delimiter character use to split the choices")

  def setChoicesDelimiter(value: String): this.type = set(choicesDelimiter, value)

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true,
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
    val suffix = "_roberta_classification"

    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          RoBertaForMultipleChoice.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          RoBertaForMultipleChoice.openvinoFile)
    }

  }

}

trait ReadablePretrainedRoBertaForMCModel
    extends ParamsAndFeaturesReadable[RoBertaForMultipleChoice]
    with HasPretrained[RoBertaForMultipleChoice] {
  override val defaultModelName: Some[String] = Some("roberta_base_qa_squad2")

  /** Java compliant-overrides */
  override def pretrained(): RoBertaForMultipleChoice = super.pretrained()

  override def pretrained(name: String): RoBertaForMultipleChoice = super.pretrained(name)

  override def pretrained(name: String, lang: String): RoBertaForMultipleChoice =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): RoBertaForMultipleChoice =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadRoBertaForMultipleChoiceDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[RoBertaForMultipleChoice] =>

  override val onnxFile: String = "roberta_mc_classification_onnx"
  override val openvinoFile: String = "roberta_mc_classification_openvino"

  def readModel(instance: RoBertaForMultipleChoice, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(
            path,
            spark,
            "roberta_mc_classification_onnx",
            zipped = true,
            useBundle = false,
            None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "roberta_mc_classification_openvino")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))

    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): RoBertaForMultipleChoice = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new RoBertaForMultipleChoice()
      .setVocabulary(vocabs)
      .setMerges(bytePairs)

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

/** This is the companion object of [[RoBertaForMultipleChoice]]. Please refer to that class for
  * the documentation.
  */
object RoBertaForMultipleChoice
    extends ReadablePretrainedRoBertaForMCModel
    with ReadRoBertaForMultipleChoiceDLModel
