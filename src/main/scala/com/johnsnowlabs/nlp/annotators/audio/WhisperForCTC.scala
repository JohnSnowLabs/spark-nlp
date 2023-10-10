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

import com.johnsnowlabs.ml.ai.Whisper
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
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
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.audio.feature_extractor.{Preprocessor, WhisperPreprocessor}
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.util.Version
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Whisper Model with a language modeling head on top for Connectionist Temporal Classification
  * (CTC).
  *
  * Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of
  * multilingual and multitask supervised data collected from the web. It transcribe in multiple
  * languages, as well as translate from those languages into English.
  *
  * The audio needs to be provided pre-processed an array of floats.
  *
  * For multilingual models, the language and the task (transcribe or translate) can be set with
  * `setLanguage` and `setTask`.
  *
 * Note that at the moment, this annotator only supports greedy search and only Spark Versions
 * 3.4 and up are supported.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val speechToText = WhisperForCTC.pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  * }}}
  * The default model is `"asr_whisper_tiny_opt"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/audio/WhisperForCTCTest.scala WhisperForCTCTestSpec]].
  *
  * '''References:'''
  *
  * [[https://arxiv.org/abs/2212.04356 Robust Speech Recognition via Large-Scale Weak Supervision]]
  *
  * '''Paper Abstract:'''
  *
  * ''We study the capabilities of speech processing systems trained simply to predict large
  * amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual
  * and multitask supervision, the resulting models generalize well to standard benchmarks and are
  * often competitive with prior fully supervised results but in a zero- shot transfer setting
  * without the need for any fine- tuning. When compared to humans, the models approach their
  * accuracy and robustness. We are releasing models and inference code to serve as a foundation
  * for further work on robust speech processing.''
  *
  * ==Example==
  *
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotators._
  * import com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC
  * import org.apache.spark.ml.Pipeline
  *
  * val audioAssembler: AudioAssembler = new AudioAssembler()
  *   .setInputCol("audio_content")
  *   .setOutputCol("audio_assembler")
  *
  * val speechToText: WhisperForCTC = WhisperForCTC
  *   .pretrained()
  *   .setInputCols("audio_assembler")
  *   .setOutputCol("text")
  *
  * val pipeline: Pipeline = new Pipeline().setStages(Array(audioAssembler, speechToText))
  *
  * val bufferedSource =
  *   scala.io.Source.fromFile("src/test/resources/audio/txt/librispeech_asr_0.txt")
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
  * |[ Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.]|
  * +------------------------------------------------------------------------------------------+
  * }}}
  *
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
class WhisperForCTC(override val uid: String)
    extends AnnotatorModel[WhisperForCTC]
    with HasBatchedAnnotateAudio[WhisperForCTC]
    with HasAudioFeatureProperties
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine
    with HasGeneratorProperties
    with HasProtectedParams {

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.AUDIO)

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WhisperForCTC"))

  /** Optional language to set for the transcription. The imported model needs to support multiple
    * languages.
    * @group param
    */
  val language =
    new Param[String](
      this,
      "language",
      "Optional parameter to set the language for the transcription.")

  /** Sets the language for the audio, formatted to e.g. `<|en|>`. Check the model description for
    * supported languages.
    *
    * @group setParam
    */
  def setLanguage(value: String): this.type = {
    require(getIsMultilingual, "Only multilingual models can have the language set.")
    require(
      value.length == 6 && value.startsWith("<|") && value.endsWith("|>"),
      "The language does not have the right format." +
        " Should be a two letter code enclosed in angle brackets with a vertical line (e.g. <|en|>).")
    require(getModelIfNotSet.tokenInVocabulary(value), "Language was not found in vocabulary.")
    set(language, value)
    this
  }

  /** @group getParam */
  def getLanguage: Option[String] = get(this.language)

  /** Sets the formatted task for the audio. Either `<|translate|>` or `<|transcribe|>`.
    *
    * Only multilingual models can do translation.
    *
    * @group setParam
    */
  override def setTask(value: String): this.type = {
    require(
      getIsMultilingual,
      "Only multilingual models can have tasks. For single language models, the default task will be transcribe.")
    require(
      value == "<|translate|>" || value == "<|transcribe|>",
      "Task should be either '<|translate|>' or '<|transcribe|>'")
    set(task, value)
    this
  }

  /** Whether or not the model is multilingual.
    *
    * @group param
    */
  val isMultilingual: ProtectedParam[Boolean] =
    new BooleanParam(this, "isMultilingual", "Whether or not the model is multilingual.")
      .setProtected()

  /** @group setParam */
  def setIsMultilingual(value: Boolean): this.type = {
    set(isMultilingual, value)
    this
  }

  /** @group getParam */
  def getIsMultilingual: Boolean = getOrDefault(this.isMultilingual)

  /** It contains TF model signatures for the loaded saved model
    *
    * @group param
    */
  val signatures: MapFeature[AnnotatorType, AnnotatorType] =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

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
  def setConfigProtoBytes(bytes: Array[Int]): this.type =
    set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Vocabulary used to encode the words to ids */
  protected[nlp] val vocabulary: MapFeature[String, Int] =
    new MapFeature(this, "vocabulary").setProtected()

  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  def getVocabulary: Map[String, Int] = $$(vocabulary)

  protected[nlp] val addedSpecialTokens: MapFeature[String, Int] =
    new MapFeature(this, "addedSpecialTokens").setProtected()

  protected[nlp] def setAddedSpecialTokens(value: Map[String, Int]): this.type =
    set(addedSpecialTokens, value)

  protected[nlp] def getAddedSpecialTokens: Map[String, Int] = $$(addedSpecialTokens)

  protected[nlp] val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  protected[nlp] def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  protected[nlp] def getGenerationConfig: GenerationConfig = $$(generationConfig)

  protected[nlp] val preprocessor: StructFeature[WhisperPreprocessor] =
    new StructFeature(this, "preprocessor").setProtected()

  protected[nlp] def setPreprocessor(value: WhisperPreprocessor): this.type =
    set(preprocessor, value)

  protected[nlp] def getPreprocessor: WhisperPreprocessor = $$(preprocessor)

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 448,
    doSample -> false,
    temperature -> 1.0,
    topK -> 1,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    batchSize -> 2,
    beamSize -> 1,
    nReturnSequences -> 1,
    isMultilingual -> true)

  private var _model: Option[Broadcast[Whisper]] = None

  /** @group getParam */
  def getModelIfNotSet: Whisper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrappers: Option[EncoderDecoderWrappers]): this.type = {
    if (_model.isEmpty) {
      val preprocessor = getPreprocessor

      _model = Some(
        spark.sparkContext.broadcast(
          new Whisper(
            tensorflowWrapper,
            onnxWrappers,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            preprocessor = preprocessor,
            vocabulary = getVocabulary,
            addedSpecialTokens = $$(addedSpecialTokens),
            generationConfig = getGenerationConfig)))
    }
    this
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          WhisperForCTC.suffix,
          WhisperForCTC.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers.get
        writeOnnxModels(
          path,
          spark,
          Seq(
            (wrappers.encoder, "encoder_model"),
            (wrappers.decoder, "decoder_model"),
            (wrappers.decoderWithPast, "decoder_with_past_model")),
          WhisperForCTC.suffix)
    }

  }

  /** Takes audio annotations and produces transcribed document annotations.
    *
    * @param batchedAnnotations
    *   Audio annotations in batches
    * @return
    *   Transcribed audio as DOCUMENT type annotation
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationAudio]]): Seq[Seq[Annotation]] = {
    batchedAnnotations.map { audioAnnotations =>
      if (audioAnnotations.nonEmpty) {
        getModelIfNotSet.generateFromAudio(
          batchAudio = audioAnnotations,
          batchSize = getBatchSize,
          maxOutputLength = getMaxOutputLength,
          minOutputLength = getMinOutputLength,
          doSample = getDoSample,
          beamSize = getBeamSize,
          numReturnSequences = getNReturnSequences,
          temperature = getTemperature,
          topK = getTopK,
          topP = getTopP,
          repetitionPenalty = getRepetitionPenalty,
          noRepeatNgramSize = getNoRepeatNgramSize,
          randomSeed = getRandomSeed,
          task = getTask,
          language = getLanguage)
      } else Seq.empty
    }
  }

}

trait ReadablePretrainedWhisperForCTCModel
    extends ParamsAndFeaturesReadable[WhisperForCTC]
    with HasPretrained[WhisperForCTC] {
  override val defaultModelName: Some[String] = Some("asr_whisper_tiny_opt")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): WhisperForCTC = super.pretrained()

  override def pretrained(name: String): WhisperForCTC = super.pretrained(name)

  override def pretrained(name: String, lang: String): WhisperForCTC =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): WhisperForCTC =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadWhisperForCTCDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[WhisperForCTC] =>

  override val tfFile: String = "whisper_ctc_tensorflow"
  override val onnxFile: String = "whisper_ctc_onnx"
  val suffix: String = "_whisper_ctc"

  private def checkVersion(spark: SparkSession): Unit = {
    val version = Version.parse(spark.version).toFloat
    require(version >= 3.4, "WhisperForCTC requires Spark versions 3.4 and up.")
  }
  def readModel(instance: WhisperForCTC, path: String, spark: SparkSession): Unit = {
    checkVersion(spark)

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(
          path,
          spark,
          WhisperForCTC.suffix,
          savedSignatures = instance.getSignatures)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val wrappers =
          readOnnxModels(
            path,
            spark,
            Seq("encoder_model", "decoder_model", "decoder_with_past_model"),
            WhisperForCTC.suffix)

        val onnxWrappers = EncoderDecoderWrappers(
          wrappers("encoder_model"),
          decoder = wrappers("decoder_model"),
          decoderWithPast = wrappers("decoder_with_past_model"))

        instance.setModelIfNotSet(spark, None, Some(onnxWrappers))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): WhisperForCTC = {
    checkVersion(spark)

    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s

    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isEncoderDecoder = true, withPast = true)

    val ppJsonString: String = loadJsonStringAsset(localModelPath, "preprocessor_config.json")

    val preprocessor: WhisperPreprocessor =
      Preprocessor.loadPreprocessorConfig(ppJsonString).asInstanceOf[WhisperPreprocessor]

    val addedTokens: Map[String, Int] =
      try {
        parse(loadJsonStringAsset(localModelPath, "added_tokens.json")).values
          .asInstanceOf[Map[String, BigInt]]
          .map {
            case (key, value) if value.isValidInt => (key, value.toInt)
            case _ =>
              throw new IllegalArgumentException(
                "Could not convert BigInt to Int while parsing added_tokens.json")
          }
      } catch {
        case _: IllegalArgumentException =>
          Map.empty
      }

    val vocabMap: Map[String, Int] = {
      val vocabJsonContent = loadJsonStringAsset(localModelPath, "vocab.json")
      parse(vocabJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]
        .map {
          case (key, value) if value.isValidInt => (key, value.toInt)
          case _ =>
            throw new IllegalArgumentException(
              "Could not convert BigInt to Int while parsing vocab.json")
        }
    }

    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val beginSuppressTokens: Array[Int] =
      (modelConfig \ "begin_suppress_tokens").extract[Array[Int]]

    val suppressTokenIds: Array[Int] =
      (modelConfig \ "suppress_tokens").extract[Array[Int]]

    val forcedDecoderIds: Array[(Int, Int)] =
      (modelConfig \ "forced_decoder_ids").extract[Array[Array[Int]]].map {
        case idxWithTokenId: Array[Int] if idxWithTokenId.length == 2 =>
          (idxWithTokenId(0), idxWithTokenId(1))
        case _ =>
          throw new Exception(
            "Could not extract forced_decoder_ids. Should be a list of tuples with 2 entries.")
      }

    val maxOutputLength = (modelConfig \ "max_length").extract[Int]
    val bosTokenId = (modelConfig \ "decoder_start_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "pad_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    // 3 means multilingual (for official models), e.g. [<|en|>, <|transcribe|>, <|notimestamps|>]
    // Single language models only force the force token to be <|notimestamps|>
    // Some custom models might have no forced tokens at all, assume its multilingual
    val isMultilingual = forcedDecoderIds.length != 1

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val annotatorModel = new WhisperForCTC()
      .setVocabulary(vocabMap)
      .setMaxOutputLength(maxOutputLength)
      .setDoNormalize(preprocessor.do_normalize)
      .setReturnAttentionMask(preprocessor.return_attention_mask)
      .setPaddingSide(preprocessor.padding_side)
      .setPaddingValue(preprocessor.padding_value)
      .setFeatureSize(preprocessor.feature_size)
      .setSamplingRate(preprocessor.sampling_rate)
      .setAddedSpecialTokens(addedTokens)
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
      .setPreprocessor(preprocessor)
      .setIsMultilingual(isMultilingual)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (tfWrapper, signatures) =
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
          .setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapperEncoder =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "encoder_model")

        val onnxWrapperDecoder =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model")

        val onnxWrapperDecoderWithPast =
          OnnxWrapper.read(
            modelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_with_past_model")

        val onnxWrappers = EncoderDecoderWrappers(
          onnxWrapperEncoder,
          onnxWrapperDecoder,
          onnxWrapperDecoderWithPast)

        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrappers))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[WhisperForCTC]]. Please refer to that class for the
  * documentation.
  */
object WhisperForCTC extends ReadablePretrainedWhisperForCTCModel with ReadWhisperForCTCDLModel
