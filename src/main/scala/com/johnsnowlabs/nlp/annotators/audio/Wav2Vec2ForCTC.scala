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

import com.johnsnowlabs.ml.ai.Wav2Vec2
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.{AUDIO, DOCUMENT}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Wav2Vec2 Model with a language modeling head on top for Connectionist Temporal Classification
  * (CTC). Wav2Vec2 was proposed in wav2vec 2.0: A Framework for Self-Supervised Learning of
  * Speech Representations by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.
  *
  * The annotator takes audio files and transcribes it as text. The audio needs to be provided
  * pre-processed an array of floats.
  *
  * Note that this annotator is currently not supported on Apple Silicon processors such as the
  * M1/M2 (Apple Silicon). This is due to the processor not supporting instructions for XLA.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val speechToText = Wav2Vec2ForCTC.pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  * }}}
  * The default model is `"asr_wav2vec2_base_960h"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/Wav2Vec2ForCTCTestSpec.scala Wav2Vec2ForCTCTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators._
  * import com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC
  * import org.apache.spark.ml.Pipeline
  *
  * val audioAssembler: AudioAssembler = new AudioAssembler()
  *   .setInputCol("audio_content")
  *   .setOutputCol("audio_assembler")
  *
  * val speechToText: Wav2Vec2ForCTC = Wav2Vec2ForCTC
  *   .pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  *
  * val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))
  *
  * val bufferedSource =
  *   scala.io.Source.fromFile("src/test/resources/audio/csv/audio_floats.csv")
  *
  * val rawFloats = bufferedSource
  *   .getLines()
  *   .map(_.split(",").head.trim.toFloat)
  *   .toArray
  * bufferedSource.close
  *
  * val processedAudioFloats = Seq(rawFloats).toDF("audio_content")
  *
  * val result = pipeline.fit(processedAudioFloats).transform(processedAudioFloats)
  * result.select("text.result").show(truncate = false)
  * +------------------------------------------------------------------------------------------+
  * |result                                                                                    |
  * +------------------------------------------------------------------------------------------+
  * |[MISTER QUILTER IS THE APOSTLE OF THE MIDLE CLASES AND WE ARE GLAD TO WELCOME HIS GOSPEL ]|
  * +------------------------------------------------------------------------------------------+
  * }}}
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
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine {

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
  val vocabulary: MapFeature[String, BigInt] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, BigInt]): this.type = set(vocabulary, value)

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

  private var _model: Option[Broadcast[Wav2Vec2]] = None

  /** @group getParam */
  def getModelIfNotSet: Wav2Vec2 = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new Wav2Vec2(
            tensorflowWrapper,
            onnxWrapper,
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
    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          "_wav_ctc",
          Wav2Vec2ForCTC.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_wav_ctc",
          Wav2Vec2ForCTC.onnxFile)
    }
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

trait ReadWav2Vec2ForAudioDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[Wav2Vec2ForCTC] =>

  override val tfFile: String = "wav_ctc_tensorflow"
  override val onnxFile: String = "wav_ctc_onnx"
  def readModel(instance: Wav2Vec2ForCTC, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tf = readTensorflowModel(path, spark, "_wav_ctc_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tf), None)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_wav_ctc_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): Wav2Vec2ForCTC = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
    val vocabJsonMap =
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]

    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    /*Universal parameters for all engines*/
    val annotatorModel = new Wav2Vec2ForCTC()
      .setVocabulary(vocabJsonMap)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setFeatureSize(preprocessorConfig.feature_size)
      .setPaddingSide(preprocessorConfig.padding_side)
      .setPaddingValue(preprocessorConfig.padding_value)
      .setReturnAttentionMask(preprocessorConfig.return_attention_mask)
      .setSamplingRate(preprocessorConfig.sampling_rate)

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
          .setModelIfNotSet(spark, Some(wrapper), None)
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[Wav2Vec2ForCTC]]. Please refer to that class for the
  * documentation.
  */
object Wav2Vec2ForCTC
    extends ReadablePretrainedWav2Vec2ForAudioModel
    with ReadWav2Vec2ForAudioDLModel
