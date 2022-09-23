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

package com.johnsnowlabs.nlp.annotators.audio

import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWav2Vec2ForCTC,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.nlp.AnnotatorType.{AUDIO, DOCUMENT}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.File
import scala.io.Source

/** Wav2Vec2 Model with a language modeling head on top for Connectionist Temporal Classification
  * (CTC). Wav2Vec2 was proposed in wav2vec 2.0: A Framework for Self-Supervised Learning of
  * Speech Representations by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.
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
class Wav2Vec2ForCTC(override val uid: String)
    extends AnnotatorModel[Wav2Vec2ForCTC]
    with HasBatchedAnnotateAudio[Wav2Vec2ForCTC]
    with HasAudioFeatureProperties
    with WriteTensorflowModel {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("Wav2Vec2ForCTC"))

  /** Output annotator type : DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** Input annotator type : AUDIO
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AUDIO)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): Wav2Vec2ForCTC.this.type =
    set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Vocabulary used to encode the words to ids
    *
    * @group param
    */
  val vocabulary: MapFeature[String, BigInt] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, BigInt]): this.type = set(vocabulary, value)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowWav2Vec2ForCTC]] = None

  /** @group getParam */
  def getModelIfNotSet: TensorflowWav2Vec2ForCTC = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowWav2Vec2ForCTC(
            tensorflow,
            configProtoBytes = getConfigProtoBytes,
            vocabs = $$(vocabulary),
            signatures = getSignatures)))
    }
    this
  }

  setDefault(batchSize -> 4)

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationAudio]]): Seq[Seq[Annotation]] = {

    // Zip annotations to the row it belongs to
    val audiosWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) => annotations.map(x => (x, i)) }

    val noneEmptyAudios = audiosWithRow.map(_._1).filter(_.result.nonEmpty).toArray

    val allAnnotations =
      if (noneEmptyAudios.nonEmpty) {
        getModelIfNotSet.predict(
          audios = noneEmptyAudios,
          batchSize = $(batchSize),
          preprocessor = Preprocessor(
            do_normalize = getDoNormalize,
            return_attention_mask = getReturnAttentionMask,
            padding_side = getPaddingSide,
            padding_value = getPaddingValue,
            feature_size = getFeatureSize,
            sampling_rate = getSamplingRate))
      } else {
        Seq.empty[Annotation]
      }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = allAnnotations
        // zip each annotation with its corresponding row index
        .zip(audiosWithRow)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowAnnotations.nonEmpty)
        rowAnnotations
      else
        Seq.empty[Annotation]
    })

  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper,
      "_wav_ctc",
      Wav2Vec2ForCTC.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedWav2Vec2ForAudioModel
    extends ParamsAndFeaturesReadable[Wav2Vec2ForCTC]
    with HasPretrained[Wav2Vec2ForCTC] {
  override val defaultModelName: Some[String] = Some("asr_wav2vec2_base_960h")

  /** Java compliant-overrides */
  override def pretrained(): Wav2Vec2ForCTC = super.pretrained()

  override def pretrained(name: String): Wav2Vec2ForCTC = super.pretrained(name)

  override def pretrained(name: String, lang: String): Wav2Vec2ForCTC =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): Wav2Vec2ForCTC =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadWav2Vec2ForAudioTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[Wav2Vec2ForCTC] =>

  override val tfFile: String = "wav_ctc_tensorflow"

  def readTensorflow(instance: Wav2Vec2ForCTC, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_wav_ctc_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): Wav2Vec2ForCTC = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath")

    val vocabPath = new File(tfModelPath + "/assets", "vocab.json")
    require(
      vocabPath.exists(),
      s"Labels file vocab.json not found in folder $tfModelPath/assets/")

    val vocabStream = ResourceHelper.getResourceStream(vocabPath.getAbsolutePath)
    val vocabJsonContent = Source.fromInputStream(vocabStream).mkString
    val vocabJsonMap =
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]

    val preprocessorConfigPath = new File(tfModelPath + "/assets", "preprocessor_config.json")
    require(
      preprocessorConfigPath.exists(),
      s"Labels file preprocessor_config.json not found in folder $tfModelPath/assets/")

    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigPath.getAbsolutePath)

    val (wrapper, signatures) =
      TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new Wav2Vec2ForCTC()
      .setVocabulary(vocabJsonMap)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setFeatureSize(preprocessorConfig.feature_size)
      .setPaddingSide(preprocessorConfig.padding_side)
      .setPaddingValue(preprocessorConfig.padding_value)
      .setReturnAttentionMask(preprocessorConfig.return_attention_mask)
      .setSamplingRate(preprocessorConfig.sampling_rate)

  }
}

/** This is the companion object of [[Wav2Vec2ForCTC]]. Please refer to that class for the
  * documentation.
  */
object Wav2Vec2ForCTC
    extends ReadablePretrainedWav2Vec2ForAudioModel
    with ReadWav2Vec2ForAudioTensorflowModel
