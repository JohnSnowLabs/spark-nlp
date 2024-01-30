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
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.M2M100
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

/** Llama 2: Open Foundation and Fine-Tuned Chat Models
  *
  * The Llama 2 release introduces a family of pretrained and fine-tuned LLMs, ranging in scale
  * from 7B to 70B parameters (7B, 13B, 70B). The pretrained models come with significant
  * improvements over the Llama 1 models, including being trained on 40% more tokens, having a
  * much longer context length (4k tokens 🤯), and using grouped-query attention for fast
  * inference of the 70B model🔥!
  *
  * However, the most exciting part of this release is the fine-tuned models (Llama 2-Chat), which
  * have been optimized for dialogue applications using Reinforcement Learning from Human Feedback
  * (RLHF). Across a wide range of helpfulness and safety benchmarks, the Llama 2-Chat models
  * perform better than most open models and achieve comparable performance to ChatGPT according
  * to human evaluations.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val m2m100 = M2M100Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"m2m100-7b"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?q=m2m100 Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/M2M100TestSpec.scala M2M100TestSpec]].
  *
  * '''References:'''
  *   - [[https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/ Llama 2: Open Foundation and Fine-Tuned Chat Models]]
  *   - [[https://github.com/facebookresearch/llama]]
  *
  * '''Paper Abstract:'''
  *
  * ''In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned
  * large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our
  * fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models
  * outperform open-source chat models on most benchmarks we tested, and based on our human
  * evaluations for helpfulness and safety, may be a suitable substitute for closed-source models.
  * We provide a detailed description of our approach to fine-tuning and safety improvements of
  * Llama 2-Chat in order to enable the community to build on our work and contribute to the
  * responsible development of LLMs.''
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.M2M100Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val m2m100 = M2M100Transformer.pretrained("m2m100-7b")
  *   .setInputCols(Array("documents"))
  *   .setMinOutputLength(10)
  *   .setMaxOutputLength(50)
  *   .setDoSample(false)
  *   .setTopK(50)
  *   .setNoRepeatNgramSize(3)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, m2m100))
  *
  * val data = Seq(
  *   "My name is Leonardo."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |result                                                                                                                                                                                              |
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |[ My name is Leonardo. I am a man of letters. I have been a man for many years. I was born in the year 1776. I came to the United States in 1776, and I have lived in the United Kingdom since 1776]|
  * +----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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
class M2M100Transformer(override val uid: String)
    extends AnnotatorModel[M2M100Transformer]
    with HasBatchedAnnotate[M2M100Transformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with HasGeneratorProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("M2M100TRANSFORMER"))

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

  /** @group setParam */
  def setRandomSeed(value: Int): M2M100Transformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** A list of token ids which are ignored in the decoder's output (Default: `Array()`)
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** Source Language (Default: `en`)
    * @group param
    */
  var srcLang = new Param[String](this, "srcLang", "Source language")

  /** Target Language (Default: `fr`)
    * @group param
    */
  var tgtLang = new Param[String](this, "tgtLang", "Target language")

  def setSrcLang(value: String): M2M100Transformer.this.type = {
    val valueLower = value.toLowerCase
    // check if language is supported
    if (!languageIds.contains(valueLower)) {
      throw new IllegalArgumentException(
        s"Language $value is not supported. Supported languages are: ${languageIds.mkString(", ")}")
    }
    srcLangToken = Some(languageIds.indexOf(valueLower))
    set(srcLang, valueLower)
  }

  def setTgtLang(value: String): M2M100Transformer.this.type = {
    val valueLower = value.toLowerCase
    // check if language is supported
    if (!languageIds.contains(valueLower)) {
      throw new IllegalArgumentException(
        s"Language $value is not supported. Supported languages are: ${languageIds.mkString(", ")}")
    }
    tgtLangToken = Some(languageIds.indexOf(valueLower))
    set(tgtLang, value)
  }

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): M2M100Transformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  def getSrcLangToken: Int = srcLangToken.getOrElse(languageIds.indexOf($(srcLang)))

  def getTgtLangToken: Int = tgtLangToken.getOrElse(languageIds.indexOf($(tgtLang)))

  private var _model: Option[Broadcast[M2M100]] = None
  private var srcLangToken: Option[Int] = None
  private var tgtLangToken: Option[Int] = None

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  private val languageIds: Array[String] = Array(
    "af",
    "am",
    "ar",
    "ast",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "ilo",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lb",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu")

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: EncoderDecoderWithoutPastWrappers,
      spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new M2M100(
            onnxWrappers,
            spp = spp,
            generationConfig = getGenerationConfig,
            vocab = $$(vocabulary))))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: M2M100 = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 4096,
    srcLang -> "en",
    tgtLang -> "fr")

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

    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }
    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        batchSize = $(batchSize),
        minOutputLength = $(minOutputLength),
        maxOutputLength = $(maxOutputLength),
        doSample = $(doSample),
        temperature = $(temperature),
        topK = $(topK),
        topP = $(topP),
        repetitionPenalty = $(repetitionPenalty),
        noRepeatNgramSize = $(noRepeatNgramSize),
        randomSeed = this.randomSeed,
        ignoreTokenIds = $(ignoreTokenIds),
        beamSize = $(beamSize),
        maxInputLength = $(maxInputLength),
        srcLangToken = getSrcLangToken,
        tgtLangToken = getTgtLangToken)
    } else {
      Seq()
    }
    Seq(processedAnnotations)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers
        val obj = getModelIfNotSet
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.encoder, "encoder_model.onnx")),
          M2M100Transformer.suffix)
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.decoder, "decoder_model.onnx")),
          M2M100Transformer.suffix)
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          M2M100Transformer.suffix,
          M2M100Transformer.sppFile)
    }
  }
}

trait ReadablePretrainedM2M100TransformerModel
    extends ParamsAndFeaturesReadable[M2M100Transformer]
    with HasPretrained[M2M100Transformer] {
  override val defaultModelName: Some[String] = Some("m2m100")

  /** Java compliant-overrides */
  override def pretrained(): M2M100Transformer = super.pretrained()

  override def pretrained(name: String): M2M100Transformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): M2M100Transformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): M2M100Transformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadM2M100TransformerDLModel extends ReadOnnxModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[M2M100Transformer] =>

  override val onnxFile: String = "m2m100_onnx"
  val suffix: String = "_m2m100"
  override val sppFile: String = "m2m100_spp"

  def readModel(instance: M2M100Transformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val decoderWrappers = readOnnxModels(path, spark, Seq("decoder_model"), suffix)
        val encoderWrappers = readOnnxModels(path, spark, Seq("encoder_model"), suffix)
        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            decoder = decoderWrappers("decoder_model"),
            encoder = encoderWrappers("encoder_model"))
        val spp = readSentencePieceModel(path, spark, "_m2m100_spp", sppFile)
        instance.setModelIfNotSet(spark, onnxWrappers, spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): M2M100Transformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isDecoder = true)
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val beginSuppressTokens: Array[Int] =
      (modelConfig \ "begin_suppress_tokens").extract[Array[Int]]

    val suppressTokenIds: Array[Int] =
      (modelConfig \ "suppress_tokens").extract[Array[Int]]

    val forcedDecoderIds: Array[(Int, Int)] = Array()

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    val annotatorModel = new M2M100Transformer()
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")
    val vocabulary: JValue =
      parse(loadJsonStringAsset(localModelPath, "vocab.json"))
    // convert to map
    val vocab = vocabulary.extract[Map[String, Int]]
    annotatorModel.setVocabulary(vocab)
    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
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

        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            encoder = onnxWrapperEncoder,
            decoder = onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, onnxWrappers, spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

object M2M100Transformer
    extends ReadablePretrainedM2M100TransformerModel
    with ReadM2M100TransformerDLModel
