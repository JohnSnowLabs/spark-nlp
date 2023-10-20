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

package com.johnsnowlabs.nlp.annotators.seq2seq

import ai.onnxruntime.{OrtEnvironment, OrtLoggingLevel}
import com.johnsnowlabs.ml.ai.Marian
import com.johnsnowlabs.ml.ai.marian.{
  MarianEncoderDecoder,
  OnnxMarianEncoderDecoder,
  TensorflowMarianEncoderDecoder
}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
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
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{
  BooleanParam,
  DoubleParam,
  IntArrayParam,
  IntParam,
  Param,
  StringArrayParam
}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import scala.jdk.CollectionConverters.asScalaBufferConverter

/** MarianTransformer: Fast Neural Machine Translation
  *
  * Marian is an efficient, free Neural Machine Translation framework written in pure C++ with
  * minimal dependencies. It is mainly being developed by the Microsoft Translator team. Many
  * academic (most notably the University of Edinburgh and in the past the Adam Mickiewicz
  * University in Poznań) and commercial contributors help with its development. MarianTransformer
  * uses the models trained by MarianNMT.
  *
  * It is currently the engine behind the Microsoft Translator Neural Machine Translation services
  * and being deployed by many companies, organizations and research projects.
  *
  * Note that this model only supports inputs up to 512 tokens. If you are working with longer
  * inputs, consider splitting them first. For example, you can use the SentenceDetectorDL
  * annotator to split longer texts into sentences first.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val marian = MarianTransformer.pretrained()
  *   .setInputCols("sentence")
  *   .setOutputCol("translation")
  * }}}
  * The default model is `"opus_mt_en_fr"`, default language is `"xx"` (meaning multi-lingual), if
  * no values are provided. For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Translation Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/multilingual/Translation_Marian.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/MarianTransformerTestSpec.scala MarianTransformerTestSpec]].
  *
  * '''Sources''' :
  *
  * [[https://marian-nmt.github.io/ MarianNMT at GitHub]]
  *
  * [[https://www.aclweb.org/anthology/P18-4020/ Marian: Fast Neural Machine Translation in C++]]
  *
  * '''Paper Abstract:'''
  *
  * ''We present Marian, an efficient and self-contained Neural Machine Translation framework with
  * an integrated automatic differentiation engine based on dynamic computation graphs. Marian is
  * written entirely in C++. We describe the design of the encoder-decoder framework and
  * demonstrate that a research-friendly toolkit can achieve high training and translation
  * speed.''
  *
  * '''Note:'''
  *
  * This is a very computationally expensive module especially on larger sequence. The use of an
  * accelerator such as GPU is recommended.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
  * import com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val marian = MarianTransformer.pretrained()
  *   .setInputCols("sentence")
  *   .setOutputCol("translation")
  *   .setMaxInputLength(30)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     sentence,
  *     marian
  *   ))
  *
  * val data = Seq("What is the capital of France? We should know this in french.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(translation.result) as result").show(false)
  * +-------------------------------------+
  * |result                               |
  * +-------------------------------------+
  * |Quelle est la capitale de la France ?|
  * |On devrait le savoir en français.    |
  * +-------------------------------------+
  * }}}
  *
  * @param uid
  *   required internal uid for saving annotator
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
class MarianTransformer(override val uid: String)
    extends AnnotatorModel[MarianTransformer]
    with HasBatchedAnnotate[MarianTransformer]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteSentencePieceModel
    with HasEngine
    with HasProtectedParams {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("MARIAN_TRANSFORMER"))

  /** Input Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Vocabulary used to encode and decode piece tokens generated by SentencePiece. This will be
    * set once the model is created and cannot be changed afterwards
    *
    * @group param
    */
  val vocabulary = new StringArrayParam(
    this,
    "vocabulary",
    "Vocabulary used to encode and decode piece words generated by SentencePiece")

  /** @group setParam */
  def setVocabulary(value: Array[String]): this.type = {
    set(vocabulary, value)
  }

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  /** Controls the maximum length for encoder inputs (source language texts) (Default: `40`)
    *
    * @group param
    */
  val maxInputLength = new IntParam(
    this,
    "maxInputLength",
    "Controls the maximum length for encoder inputs (source language texts)")

  /** @group setParam * */
  def setMaxInputLength(value: Int): this.type = {
    require(value <= 512, "MarianTransformer model does not support sequences longer than 512.")
    set(maxInputLength, value)
    this
  }

  /** @group getParam */
  def getMaxInputLength: Int = $(maxInputLength)

  /** Controls the maximum length for decoder outputs (target language texts) (Default: `40`)
    *
    * @group param
    */
  val maxOutputLength = new IntParam(
    this,
    "maxOutputLength",
    "Controls the maximum length for decoder outputs (target language texts)")

  /** @group setParam * */
  def setMaxOutputLength(value: Int): this.type = {
    set(maxOutputLength, value)
  }

  /** @group getParam */
  def getMaxOutputLength: Int = $(maxOutputLength)

  /** A string representing the target language in the form of >>id<< (id = valid target language
    * ID) (Default: `""`)
    *
    * langId is only needed if the model generates multi-lingual target language texts. For
    * instance, for a 'en-fr' model this param is not required to be set.
    *
    * @group param
    */
  var langId = new Param[String](
    this,
    "langId",
    "A string representing the target language in the form of >>id<< (id = valid target language ID)")

  /** @group setParam */
  def setLangId(lang: String): MarianTransformer.this.type = {
    set(langId, lang)
  }

  /** @group getParam */
  def getLangId: String = $(langId)

  /** A list of token ids which are ignored in the decoder's output
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): MarianTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group getParam */
  def setConfigProtoBytes(bytes: Array[Int]): MarianTransformer.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam * */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

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

  /** Whether or not to use sampling, use greedy decoding otherwise (Default: `false`)
    *
    * @group param
    */
  val doSample = new BooleanParam(
    this,
    "doSample",
    "Whether or not to use sampling; use greedy decoding otherwise")

  /** @group setParam */
  def setDoSample(value: Boolean): this.type = {
    set(doSample, value)
    this
  }

  /** @group getParam */
  def getDoSample: Boolean = $(this.doSample)

  /** The value used to module the next token probabilities (Default: `1.0`)
    *
    * @group param
    */
  val temperature =
    new DoubleParam(this, "temperature", "The value used to module the next token probabilities")

  /** @group setParam */
  def setTemperature(value: Double): this.type = {
    set(temperature, value)
    this
  }

  /** @group getParam */
  def getTemperature: Double = $(this.temperature)

  /** The number of highest probability vocabulary tokens to keep for top-k-filtering (Default:
    * `50`)
    *
    * @group param
    */
  val topK = new IntParam(
    this,
    "topK",
    "The number of highest probability vocabulary tokens to keep for top-k-filtering")

  /** @group setParam */
  def setTopK(value: Int): this.type = {
    set(topK, value)
    this
  }

  /** @group getParam */
  def getTopK: Int = $(this.topK)

  /** If set to float < `1.0`, only the most probable tokens with probabilities that add up to
    * `topP` or higher are kept for generation (Default: `1.0`)
    *
    * @group param
    */
  val topP = new DoubleParam(
    this,
    "topP",
    "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation")

  /** @group setParam */
  def setTopP(value: Double): this.type = {
    set(topP, value)
    this
  }

  /** @group getParam */
  def getTopP: Double = $(this.topP)

  /** The parameter for repetition penalty (Default: `1.0`). `1.0` means no penalty. See
    * [[https://arxiv.org/pdf/1909.05858.pdf this paper]] for more details.
    *
    * @group param
    */
  val repetitionPenalty = new DoubleParam(
    this,
    "repetitionPenalty",
    "The parameter for repetition penalty. 1.0 means no penalty.")

  /** @group setParam */
  def setRepetitionPenalty(value: Double): this.type = {
    set(repetitionPenalty, value)
    this
  }

  /** @group getParam */
  def getRepetitionPenalty: Double = $(this.repetitionPenalty)

  /** If set to int > `0`, all ngrams of that size can only occur once (Default: `0`)
    *
    * @group param
    */
  val noRepeatNgramSize = new IntParam(
    this,
    "noRepeatNgramSize",
    "If set to int > 0, all ngrams of that size can only occur once")

  /** @group setParam */
  def setNoRepeatNgramSize(value: Int): this.type = {
    set(noRepeatNgramSize, value)
    this
  }

  /** @group getParam */
  def getNoRepeatNgramSize: Int = $(this.noRepeatNgramSize)

  /** Optional Random seed for the model. Needs to be of type `Long`.
    *
    * @group param
    */
  var randomSeed: Option[Long] = None

  /** @group setParam */
  def setRandomSeed(value: Long): this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** @group getParam */
  def getRandomSeed: Option[Long] = this.randomSeed

  /** The Tensorflow Marian Model */
  private var _model: Option[Broadcast[MarianEncoderDecoder]] = None

  /** @group setParam * */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflow: TensorflowWrapper,
      sppSrc: SentencePieceWrapper,
      sppTrg: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowMarianEncoderDecoder(
            tensorflow,
            sppSrc,
            sppTrg,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures)))
    }
    this
  }

  def setModelIfNotSet(
      spark: SparkSession,
      encoder: OnnxWrapper,
      decoder: OnnxWrapper,
      sppSrc: SentencePieceWrapper,
      sppTrg: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new OnnxMarianEncoderDecoder(encoder, decoder, sppSrc, sppTrg)))
    }
    this
  }

  /** @group setParam * */
  def getModelIfNotSet: MarianEncoderDecoder = _model.get.value

  def getVocabulary: Array[String] = {
    if ($(vocabulary).isInstanceOf[java.util.ArrayList[String]]) {
      val arrayListValue = $(vocabulary).asInstanceOf[java.util.ArrayList[String]]
      arrayListValue.asScala.toArray
    } else $(vocabulary)
  }

  setDefault(
    maxInputLength -> 40,
    maxOutputLength -> 40,
    batchSize -> 1,
    langId -> "",
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    ignoreTokenIds -> Array(),
    batchSize -> 1)

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

    val nonEmptyBatch = batchedAnnotations.filter(_.nonEmpty)

    val allAnnotations = nonEmptyBatch.zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }

    val processedAnnotations = if (allAnnotations.nonEmpty) {

      this.getModelIfNotSet
        .predict(
          sentences = allAnnotations.map(_._1),
          maxInputLength = getMaxInputLength,
          maxOutputLength = getMaxOutputLength,
          vocabs = getVocabulary,
          langId = getLangId,
          batchSize = getBatchSize,
          doSample = getDoSample,
          temperature = getTemperature,
          topK = getTopK,
          topP = getTopP,
          repetitionPenalty = getRepetitionPenalty,
          noRepeatNgramSize = getNoRepeatNgramSize,
          randomSeed = getRandomSeed,
          ignoreTokenIds = getIgnoreTokenIds)
        .toSeq
    } else {
      Seq()
    }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = processedAnnotations
        // zip each annotation with its corresponding row index
        .zip(allAnnotations)
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

    getModelIfNotSet match {
      case obj: OnnxMarianEncoderDecoder =>
        writeOnnxModel(path, spark, obj.onnxEncoder, "", MarianTransformer.onnxEncoderFile)
        writeOnnxModel(path, spark, obj.onnxDecoder, "", MarianTransformer.onnxDecoderFile)
        writeSentencePieceModel(
          path,
          spark,
          obj.sppSrc,
          "_src_marian",
          MarianTransformer.sppFile + "_src")
        writeSentencePieceModel(
          path,
          spark,
          obj.sppTrg,
          "_trg_marian",
          MarianTransformer.sppFile + "_trg")

      case obj: TensorflowMarianEncoderDecoder =>
        writeTensorflowModelV2(
          path,
          spark,
          obj.tensorflow,
          "_t5",
          T5Transformer.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
        writeSentencePieceModel(
          path,
          spark,
          obj.sppSrc,
          "_src_marian",
          MarianTransformer.sppFile + "_src")
        writeSentencePieceModel(
          path,
          spark,
          obj.sppTrg,
          "_trg_marian",
          MarianTransformer.sppFile + "_trg")
    }
  }

}

trait ReadablePretrainedMarianMTModel
    extends ParamsAndFeaturesReadable[MarianTransformer]
    with HasPretrained[MarianTransformer] {
  override val defaultModelName: Some[String] = Some("opus_mt_en_fr")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): MarianTransformer = super.pretrained()

  override def pretrained(name: String): MarianTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): MarianTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MarianTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMarianMTDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel
    with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[MarianTransformer] =>

  override val tfFile: String = "marian_tensorflow"
  override val sppFile: String = "marian_spp"
  val onnxEncoderFile: String = "encoder.onxx"
  val onnxDecoderFile: String = "decoder.onxx"
  override val onnxFile: String = ""

  def readModel(instance: MarianTransformer, path: String, spark: SparkSession): Unit = {
    val sppSrc = readSentencePieceModel(path, spark, "_src_marian", sppFile + "_src")
    val sppTrg = readSentencePieceModel(path, spark, "_trg_marian", sppFile + "_trg")

    instance.getEngine match {
      case ONNX.name =>
        OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
        val onnxModels =
          readOnnxModels(path, spark, Seq(onnxEncoderFile, onnxDecoderFile), suffix = "")
        instance
          .setModelIfNotSet(
            spark,
            onnxModels(onnxEncoderFile),
            onnxModels(onnxDecoderFile),
            sppSrc,
            sppTrg)
      case _ =>
        val tf =
          readTensorflowModel(path, spark, "_marian_tf", savedSignatures = instance.getSignatures)
        instance.setModelIfNotSet(spark, tf, sppSrc, sppTrg)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MarianTransformer = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath, isEncoderDecoder = true)

    val sppSrc = loadSentencePieceAsset(localModelPath, "source.spm")
    val sppTrg = loadSentencePieceAsset(localModelPath, "target.spm")
    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap.toSeq
      .sortBy(_._2)
      .map(x => x._1.mkString)
      .toArray

    /*Universal parameters for all engines*/
    val annotatorModel = new MarianTransformer()
      .setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) = TensorflowWrapper.read(
          localModelPath,
          zipped = false,
          useBundle = true,
          tags = Array("serve"))

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setEngine(TensorFlow.name)
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, wrapper, sppSrc, sppTrg)
      case ONNX.name =>
        OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)

        val onnxEncoder = OnnxWrapper.read(
          modelPath,
          modelName = "encoder_model",
          zipped = false,
          useBundle = true)

        val onnxDecoder = OnnxWrapper.read(
          modelPath,
          modelName = "decoder_model_merged",
          zipped = false,
          useBundle = true)

        annotatorModel
          .setEngine(ONNX.name)
          .setModelIfNotSet(spark, onnxEncoder, onnxDecoder, sppSrc, sppTrg)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MarianTransformer]]. Please refer to that class for the
  * documentation.
  */
object MarianTransformer
    extends ReadablePretrainedMarianMTModel
    with ReadMarianMTDLModel
    with ReadSentencePieceModel
