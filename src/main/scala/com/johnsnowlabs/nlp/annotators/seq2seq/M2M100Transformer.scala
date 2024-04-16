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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.util.FileHelper
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** M2M100 : multilingual translation model
  *
  * M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
  * multilingual translation.
  *
  * The model can directly translate between the 9,900 directions of 100 languages.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val m2m100 = M2M100Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"m2m100_418M"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?q=m2m100 Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/M2M100TestSpec.scala M2M100TestSpec]].
  *
  * '''References:'''
  *   - [[https://arxiv.org/pdf/2010.11125.pdf Beyond English-Centric Multilingual Machine Translation]]
  *   - [[https://github.com/pytorch/fairseq/tree/master/examples/m2m_100]]
  *
  * '''Paper Abstract:'''
  *
  * ''Existing work in translation demonstrated the potential of massively multilingual machine
  * translation by training a single model able to translate between any pair of languages.
  * However, much of this work is English-Centric by training only on data which was translated
  * from or to English. While this is supported by large sources of training data, it does not
  * reflect translation needs worldwide. In this work, we create a true Many-to-Many multilingual
  * translation model that can translate directly between any pair of 100 languages. We build and
  * open source a training dataset that covers thousands of language directions with supervised
  * data, created through large-scale mining. Then, we explore how to effectively increase model
  * capacity through a combination of dense scaling and language-specific sparse parameters to
  * create high quality models. Our focus on non-English-Centric models brings gains of more than
  * 10 BLEU when directly translating between non-English directions while performing
  * competitively to the best single systems of WMT. We open-source our scripts so that others may
  * reproduce the data, evaluation, and final M2M-100 model.''
  *
  * '''Languages Covered:'''
  *
  * Afrikaans (af), Amharic (am), Arabic (ar), Asturian (ast), Azerbaijani (az), Bashkir (ba),
  * Belarusian (be), Bulgarian (bg), Bengali (bn), Breton (br), Bosnian (bs), Catalan; Valencian
  * (ca), Cebuano (ceb), Czech (cs), Welsh (cy), Danish (da), German (de), Greeek (el), English
  * (en), Spanish (es), Estonian (et), Persian (fa), Fulah (ff), Finnish (fi), French (fr),
  * Western Frisian (fy), Irish (ga), Gaelic; Scottish Gaelic (gd), Galician (gl), Gujarati (gu),
  * Hausa (ha), Hebrew (he), Hindi (hi), Croatian (hr), Haitian; Haitian Creole (ht), Hungarian
  * (hu), Armenian (hy), Indonesian (id), Igbo (ig), Iloko (ilo), Icelandic (is), Italian (it),
  * Japanese (ja), Javanese (jv), Georgian (ka), Kazakh (kk), Central Khmer (km), Kannada (kn),
  * Korean (ko), Luxembourgish; Letzeburgesch (lb), Ganda (lg), Lingala (ln), Lao (lo), Lithuanian
  * (lt), Latvian (lv), Malagasy (mg), Macedonian (mk), Malayalam (ml), Mongolian (mn), Marathi
  * (mr), Malay (ms), Burmese (my), Nepali (ne), Dutch; Flemish (nl), Norwegian (no), Northern
  * Sotho (ns), Occitan (post 1500) (oc), Oriya (or), Panjabi; Punjabi (pa), Polish (pl), Pushto;
  * Pashto (ps), Portuguese (pt), Romanian; Moldavian; Moldovan (ro), Russian (ru), Sindhi (sd),
  * Sinhala; Sinhalese (si), Slovak (sk), Slovenian (sl), Somali (so), Albanian (sq), Serbian
  * (sr), Swati (ss), Sundanese (su), Swedish (sv), Swahili (sw), Tamil (ta), Thai (th), Tagalog
  * (tl), Tswana (tn), Turkish (tr), Ukrainian (uk), Urdu (ur), Uzbek (uz), Vietnamese (vi), Wolof
  * (wo), Xhosa (xh), Yiddish (yi), Yoruba (yo), Chinese (zh), Zulu (zu)
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
  * val m2m100 = M2M100Transformer.pretrained("m2m100_418M")
  *   .setInputCols(Array("documents"))
  *   .setSrcLang("zh")
  *   .serTgtLang("en")
  *   .setMaxOutputLength(100)
  *   .setDoSample(false)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, m2m100))
  *
  * val data = Seq(
  *   "生活就像一盒巧克力。"
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +-------------------------------------------------------------------------------------------+
  * |result                                                                                     |
  * +-------------------------------------------------------------------------------------------+
  * |[ Life is like a box of chocolate.]                                                        |
  * +-------------------------------------------------------------------------------------------+
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
    minOutputLength -> 10,
    maxOutputLength -> 200,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 1024,
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
  override val defaultModelName: Some[String] = Some("m2m100_418M")
  override val defaultLang: String = "xx"

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
        val decoderWrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val encoderWrappers =
          readOnnxModels(path, spark, Seq("encoder_model.onnx"), suffix)
        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            decoder = decoderWrappers("decoder_model.onnx"),
            encoder = encoderWrappers("encoder_model.onnx"))
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
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "encoder_model",
            onnxFileSuffix = None)
        val onnxWrapperDecoder =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model",
            onnxFileSuffix = None)

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
