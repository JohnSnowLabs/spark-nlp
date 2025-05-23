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

package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.Florence2
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Florence2Wrappers
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.Openvino
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Florence2 : multilingual translation model
  *
  * Florence2 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
  * multilingual translation.
  *
  * The model can directly translate between the 9,900 directions of 100 languages.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val Florence2 = Florence2Transformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"Florence2_418M"`, if no name is provided. For available pretrained
  * models please see the [[https://sparknlp.org/models?q=Florence2 Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/Florence2TestSpec.scala Florence2TestSpec]].
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
  * import com.johnsnowlabs.nlp.annotators.seq2seq.Florence2Transformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val Florence2 = Florence2Transformer.pretrained("Florence2_418M")
  *   .setInputCols(Array("documents"))
  *   .setSrcLang("zh")
  *   .serTgtLang("en")
  *   .setMaxOutputLength(100)
  *   .setDoSample(false)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, Florence2))
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
class Florence2Transformer(override val uid: String)
    extends AnnotatorModel[Florence2Transformer]
    with HasBatchedAnnotateImage[Florence2Transformer]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  def this() = this(Identifiable.randomUID("Florence2TRANSFORMER"))

  /** Input annotator type : DOCUMENT
    *
    * @group param
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): Florence2Transformer.this.type = {
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

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): Florence2Transformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  private var _model: Option[Broadcast[Florence2]] = None

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

  /** Holding merges.txt coming from RoBERTa model
    *
    * @group param
    */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** Additional tokens to be added to the vocabulary
    *
    * @group param
    */
  val addedTokens: MapFeature[String, Int] = new MapFeature(this, "addedTokens").setProtected()

  /** @group setParam */
  def setAddedTokens(value: Map[String, Int]): this.type = set(addedTokens, value)

  /** Stop tokens to terminate the generation
    *
    * @group param
    */
  override val stopTokenIds =
    new IntArrayParam(this, "stopTokenIds", "Stop tokens to terminate the generation")

  /** @group setParam */
  override def setStopTokenIds(value: Array[Int]): this.type = {
    set(stopTokenIds, value)
  }

  /** @group getParam */
  override def getStopTokenIds: Array[Int] = $(stopTokenIds)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[Florence2Wrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Florence2(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Florence2 = _model.get.value

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
    stopTokenIds -> Array(2))

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationImage]]): Seq[Seq[Annotation]] = {

    batchedAnnotations.map { cleanAnnotationImages =>
      val validImages = cleanAnnotationImages.filter(_.result.nonEmpty)
      val questionAnnotations = extractInputAnnotation(validImages)

      getModelIfNotSet.predict(
        questionAnnotations,
        validImages.toSeq,
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
        maxInputLength = $(maxInputLength))
    }

  }

  private def extractInputAnnotation(
      annotationImages: Array[AnnotationImage]): Seq[Annotation] = {
    val questions = annotationImages.map(annotationImage => {
      val imageText =
        if (annotationImage.text.nonEmpty) annotationImage.text
        else
          "<s>Locate the objects with category name in the image.</s>" // default question
      Annotation(imageText)
    })

    questions
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case Openvino.name =>
        val wrappers = getModelIfNotSet.openvinoWrapper
        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.encoderModel, "encoder.xml")),
          PaliGemmaForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.decoderModel, "decoder.xml")),
          PaliGemmaForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbeddingsModel, "text_embedding.xml")),
          PaliGemmaForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEmbedModel, "image_embedding.xml")),
          PaliGemmaForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.modelMergerModel, "merge_model.xml")),
          PaliGemmaForMultiModal.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }
}

trait ReadablePretrainedFlorence2TransformerModel
    extends ParamsAndFeaturesReadable[Florence2Transformer]
    with HasPretrained[Florence2Transformer] {
  override val defaultModelName: Some[String] = Some("florence2_")

  /** Java compliant-overrides */
  override def pretrained(): Florence2Transformer = super.pretrained()

  override def pretrained(name: String): Florence2Transformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): Florence2Transformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): Florence2Transformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadFlorence2TransformerDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[Florence2Transformer] =>

  val suffix: String = "_Florence2"
  override val openvinoFile: String = "Florence2_openvino"

  def readModel(instance: Florence2Transformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case Openvino.name =>
        val decoderWrappers =
          readOpenvinoModels(path, spark, Seq("decoder.xml"), suffix)
        val encoderWrappers =
          readOpenvinoModels(path, spark, Seq("encoder.xml"), suffix)
        val textEmbeddingsWrappers =
          readOpenvinoModels(path, spark, Seq("text_embedding.xml"), suffix)
        val imageEmbeddingsWrappers =
          readOpenvinoModels(path, spark, Seq("image_embedding.xml"), suffix)
        val modelMergerWrappers =
          readOpenvinoModels(path, spark, Seq("merge_model.xml"), suffix)
        val ovWrapper = {
          Florence2Wrappers(
            encoderModel = encoderWrappers("encoder.xml"),
            decoderModel = decoderWrappers("decoder.xml"),
            textEmbeddingsModel = textEmbeddingsWrappers("text_embedding.xml"),
            imageEmbedModel = imageEmbeddingsWrappers("image_embedding.xml"),
            modelMergerModel = modelMergerWrappers("merge_model.xml"))
        }
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "FlorenceFeatureExtractor",
          instance.getImageMean,
          instance.getImageStd,
          instance.getResample,
          instance.getSize)
        instance.setModelIfNotSet(spark, preprocessor, None, Some(ovWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): Florence2Transformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s
    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom =
          Some(List("encoder", "decoder", "text_embedding", "merge_model", "image_embedding")))
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig = Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)
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

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "pad_token_id").extract[Int]
    val vocabSize = (modelConfig \ "text_config" \ "vocab_size").extract[Int]

    // Check if tokenizer.json exists
    val tokenizerPath = s"$localModelPath/assets/tokenizer.json"
    val tokenizerExists = new java.io.File(tokenizerPath).exists()
    val (vocabs, addedTokens, bytePairs) = if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      var vocabs: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]

      val bytePairs = (tokenizerConfig \ "model" \ "merges")
        .extract[List[Array[String]]]
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap

      val addedTokens = (tokenizerConfig \ "added_tokens")
        .extract[List[Map[String, Any]]]
        .map { token =>
          val id = token("id").asInstanceOf[BigInt].intValue()
          val content = token("content").asInstanceOf[String]
          (content, id)
        }
        .toMap

      addedTokens.foreach { case (content, id) =>
        vocabs += (content -> id)
      }
      (vocabs, addedTokens, bytePairs)
    } else {
      val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
      val addedTokens = loadTextAsset(localModelPath, "added_tokens.txt").zipWithIndex.toMap
      val bytePairs = loadTextAsset(localModelPath, "merges.txt")
        .map(_.split(" "))
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap
      (vocabs, addedTokens, bytePairs)
    }

    val annotatorModel = new Florence2Transformer()
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
      .setVocabulary(vocabs)
      .setMerges(bytePairs)
      .setAddedTokens(addedTokens)
      .setSize(preprocessorConfig.size)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    detectedEngine match {
      case Openvino.name =>
        val openvinoEncoderWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "encoder")
        val openvinoDecoderWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "decoder")
        val openvinoTextEmbeddingsWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "text_embedding")
        val openvinoImageEmbeddingsWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "image_embedding")
        val openvinoModelMergerWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "merge_model")
        val openvinoWrapper =
          Florence2Wrappers(
            encoderModel = openvinoEncoderWrapper,
            decoderModel = openvinoDecoderWrapper,
            textEmbeddingsModel = openvinoTextEmbeddingsWrapper,
            imageEmbedModel = openvinoImageEmbeddingsWrapper,
            modelMergerModel = openvinoModelMergerWrapper)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

object Florence2Transformer
    extends ReadablePretrainedFlorence2TransformerModel
    with ReadFlorence2TransformerDLModel
