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

import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.{OrtEnvironment, OrtLoggingLevel}
import com.johnsnowlabs.ml.ai.t5.{
  OnnxT5EncoderDecoder,
  T5EncoderDecoder,
  TensorflowT5EncoderDecoder
}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** T5: the Text-To-Text Transfer Transformer
  *
  * T5 reconsiders all NLP tasks into a unified text-to-text-format where the input and output are
  * always text strings, in contrast to BERT-style models that can only output either a class
  * label or a span of the input. The text-to-text framework is able to use the same model, loss
  * function, and hyper-parameters on any NLP task, including machine translation, document
  * summarization, question answering, and classification tasks (e.g., sentiment analysis). T5 can
  * even apply to regression tasks by training it to predict the string representation of a number
  * instead of the number itself.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val t5 = T5Transformer.pretrained()
  *   .setTask("summarize:")
  *   .setInputCols("document")
  *   .setOutputCol("summaries")
  * }}}
  * The default model is `"t5_small"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?q=t5 Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/question-answering/Question_Answering_and_Summarization_with_T5.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/T5TestSpec.scala T5TestSpec]].
  *
  * '''References:'''
  *   - [[https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer]]
  *   - [[https://arxiv.org/abs/1910.10683 Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer]]
  *   - [[https://github.com/google-research/text-to-text-transfer-transformer]]
  *
  * '''Paper Abstract:'''
  *
  * ''Transfer learning, where a model is first pre-trained on a data-rich task before being
  * fine-tuned on a downstream task, has emerged as a powerful technique in natural language
  * processing (NLP). The effectiveness of transfer learning has given rise to a diversity of
  * approaches, methodology, and practice. In this paper, we explore the landscape of transfer
  * learning techniques for NLP by introducing a unified framework that converts all text-based
  * language problems into a text-to-text format. Our systematic study compares pre-training
  * objectives, architectures, unlabeled data sets, transfer approaches, and other factors on
  * dozens of language understanding tasks. By combining the insights from our exploration with
  * scale and our new Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many
  * benchmarks covering summarization, question answering, text classification, and more. To
  * facilitate future work on transfer learning for NLP, we release our data set, pre-trained
  * models, and code.''
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val t5 = T5Transformer.pretrained("t5_small")
  *   .setTask("summarize:")
  *   .setInputCols(Array("documents"))
  *   .setMaxOutputLength(200)
  *   .setOutputCol("summaries")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))
  *
  * val data = Seq(
  *   "Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a " +
  *     "downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness" +
  *     " of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this " +
  *     "paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework " +
  *     "that converts all text-based language problems into a text-to-text format. Our systematic study compares " +
  *     "pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens " +
  *     "of language understanding tasks. By combining the insights from our exploration with scale and our new " +
  *     "Colossal Clean Crawled Corpus, we achieve state-of-the-art results on many benchmarks covering " +
  *     "summarization, question answering, text classification, and more. To facilitate future work on transfer " +
  *     "learning for NLP, we release our data set, pre-trained models, and code."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("summaries.result").show(false)
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                                        |
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[transfer learning has emerged as a powerful technique in natural language processing (NLP) the effectiveness of transfer learning has given rise to a diversity of approaches, methodologies, and practice .]|
  * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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
class T5Transformer(override val uid: String)
    extends AnnotatorModel[T5Transformer]
    with HasBatchedAnnotate[T5Transformer]
    with ParamsAndFeaturesWritable
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasCaseSensitiveProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("T5TRANSFORMER"))

  /** Input annotator type : DOCUMENT
    *
    * @group param
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group param
    */
  override val outputAnnotatorType: String = DOCUMENT

  /** Set transformer task, e.g. `"summarize:"` (Default: `""`). The T5 task needs to be in the
    * format `"task:"`.
    *
    * @group param
    */
  val task = new Param[String](this, "task", "Set transformer task, e.g. 'summarize'")

  /** @group setParam */
  def setTask(value: String): T5Transformer.this.type = {
    set(task, value)
    this
  }

  /** Stop text generation when the end-of-sentence token is encountered.
    *
    * @group param
    */
  val stopAtEos =
    new BooleanParam(parent = this, name = "stopAtEos", doc = "Stop at end-of-sentence token.")

  /** Determines whether text generation stops when the end-of-sentence token is encountered.
    *
    * @group setParam
    */
  def setStopAtEos(value: Boolean): this.type = set(stopAtEos, value)

  /** Checks whether text generation stops when the end-of-sentence token is encountered.
    *
    * @group getParam
    */
  def getStopAtEos: Boolean = $(stopAtEos)

  /** Minimum length of the sequence to be generated (Default: `0`)
    *
    * @group param
    */
  val minOutputLength =
    new IntParam(this, "minOutputLength", "Minimum length of the sequence to be generated")

  /** @group setParam */
  def setMinOutputLength(value: Int): T5Transformer.this.type = {
    set(minOutputLength, value)
    this
  }

  /** @group getParam */
  def getMinOutputLength: Int = $(this.minOutputLength)

  /** Maximum length of the sequence to be generated (Default: `20`)
    *
    * @group param
    */
  val maxOutputLength =
    new IntParam(this, "maxOutputLength", "Maximum length of the sequence to be generated")

  /** @group setParam */
  def setMaxOutputLength(value: Int): T5Transformer.this.type = {
    set(maxOutputLength, value)
    this
  }

  /** @group getParam */
  def getMaxOutputLength: Int = $(this.maxOutputLength)

  /** ML framework type
    *
    * @group param
    */
  val mlFrameworkType =
    new Param[String](parent = this, name = "mlFrameworkType", doc = "ML framework (TF, ONNX)")

  /** Set ML framework type
    *
    * @group setParam
    */
  def setMlFrameworkType(value: String): this.type = {
    set(mlFrameworkType, value)
    this
  }

  /** Get ML framework type
    *
    * @group getParam
    */
  def getMlFrameworkType: String = $(mlFrameworkType)

  /** Cache internal state of the model to improve performance. This param can only be set when
    * importing the model.
    *
    * @group param
    */
  private[johnsnowlabs] val useCache =
    new BooleanParam(parent = this, name = "useCache", doc = "Cache internal state of the model")

  private[johnsnowlabs] def setUseCache(value: Boolean): this.type = {
    set(useCache, value)
    this
  }

  private[johnsnowlabs] def getUseCache: Boolean = $(useCache)

  /** Maximum number of new tokens to be generated (Default: `20`)
    *
    * @group param
    */
  val maxNewTokens =
    new IntParam(this, "maxNewTokens", "Maximum number of new tokens to be generated")

  /** @group setParam */
  def setMaxNewTokens(value: Int): T5Transformer.this.type = {
    set(maxNewTokens, value)
    this
  }

  /** @group getParam */
  def getMaxNewTokens: Int = $(this.maxNewTokens)

  /** Whether or not to use sampling, use greedy decoding otherwise (Default: `false`)
    *
    * @group param
    */
  val doSample = new BooleanParam(
    this,
    "doSample",
    "Whether or not to use sampling; use greedy decoding otherwise")

  /** @group setParam */
  def setDoSample(value: Boolean): T5Transformer.this.type = {
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
  def setTemperature(value: Double): T5Transformer.this.type = {
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
  def setTopK(value: Int): T5Transformer.this.type = {
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
  def setTopP(value: Double): T5Transformer.this.type = {
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
  def setRepetitionPenalty(value: Double): T5Transformer.this.type = {
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
  def setNoRepeatNgramSize(value: Int): T5Transformer.this.type = {
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
  def setRandomSeed(value: Long): T5Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** @group getParam */
  def getRandomSeed: Option[Long] = this.randomSeed

  /** A list of token ids which are ignored in the decoder's output
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): T5Transformer.this.type = {
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

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): T5Transformer.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
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

  private var _model: Option[Broadcast[T5EncoderDecoder]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tfWrapper: TensorflowWrapper,
      spp: SentencePieceWrapper,
      useCache: Boolean): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowT5EncoderDecoder(
            tensorflow = tfWrapper,
            spp = spp,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            useCache = useCache)))
    }
    this
  }

  def setModelIfNotSet(
      spark: SparkSession,
      encoder: OnnxWrapper,
      decoder: OnnxWrapper,
      spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(new OnnxT5EncoderDecoder(encoder, decoder, spp)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: T5EncoderDecoder = _model.get.value

  setDefault(
    task -> "",
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 0,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    stopAtEos -> true,
    mlFrameworkType -> TensorFlow.name,
    maxNewTokens -> 512,
    useCache -> false)

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }

    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        task = $(task),
        batchSize = $(batchSize),
        maxTextLength = $(maxOutputLength),
        maxNewTokens = $(maxNewTokens),
        stopAtEos = $(stopAtEos),
        doSample = $(doSample),
        topK = $(topK),
        topP = $(topP),
        temperature = $(temperature),
        noRepeatNgramSize = $(noRepeatNgramSize),
        repetitionPenalty = $(repetitionPenalty),
        ignoreTokenIds = $(ignoreTokenIds),
        isCaseSensitive = $(caseSensitive))
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
      case obj: OnnxT5EncoderDecoder =>
        writeOnnxModel(path, spark, obj.onnxEncoder, "", T5Transformer.onnxEncoderFile)
        writeOnnxModel(path, spark, obj.onnxDecoder, "", T5Transformer.onnxDecoderFile)
        writeSentencePieceModel(path, spark, obj.spp, "_med_seq2seq", T5Transformer.sppFile)

      case obj: TensorflowT5EncoderDecoder =>
        writeTensorflowModelV2(
          path,
          spark,
          obj.tensorflow,
          "_t5",
          T5Transformer.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
        writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_t5", T5Transformer.sppFile)
    }
  }
}

trait ReadablePretrainedT5TransformerModel
    extends ParamsAndFeaturesReadable[T5Transformer]
    with HasPretrained[T5Transformer] {
  override val defaultModelName: Some[String] = Some("t5_small")

  /** Java compliant-overrides */
  override def pretrained(): T5Transformer = super.pretrained()

  override def pretrained(name: String): T5Transformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): T5Transformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): T5Transformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadT5TransformerDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel
    with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[T5Transformer] =>

  override val tfFile: String = "t5_tensorflow"
  override val sppFile: String = "t5_spp"

  val onnxEncoderFile: String = "encoder.onxx"
  val onnxDecoderFile: String = "decoder.onxx"

  override val onnxFile: String = ""

  def readModel(instance: T5Transformer, path: String, spark: SparkSession): Unit = {

    val spp = readSentencePieceModel(path, spark, "_t5_spp", sppFile)

    instance.getMlFrameworkType.toLowerCase match {
      case ONNX.name =>
        OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
        val onnxModels = readOnnxModels(
          path,
          spark,
          Seq(onnxEncoderFile, onnxDecoderFile),
          suffix = "")
        instance
          .setModelIfNotSet(spark, onnxModels(onnxEncoderFile), onnxModels(onnxDecoderFile), spp)
      case _ =>
        val tf = readTensorflowModel(
          path,
          spark,
          "_t5_tf",
          initAllTables = false,
          savedSignatures = instance.getSignatures)
        instance.setModelIfNotSet(spark, tf, spp, instance.getUseCache)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): T5Transformer = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath, isEncoderDecoder = true)

    /*Universal parameters for all engines*/
    val annotatorModel = new T5Transformer()

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    val spModel = loadSentencePieceAsset(localModelPath, "spiece.model")

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) = TensorflowWrapper.read(
          localModelPath,
          zipped = false,
          useBundle = true,
          tags = Array("serve"),
          initAllTables = false)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, wrapper, spModel, useCache = false)

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
          .setMlFrameworkType(ONNX.name)
          .setModelIfNotSet(spark, onnxEncoder, onnxDecoder, spModel)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[T5Transformer]]. Please refer to that class for the
  * documentation.
  */
object T5Transformer
    extends ReadablePretrainedT5TransformerModel
    with ReadT5TransformerDLModel
    with ReadSentencePieceModel
