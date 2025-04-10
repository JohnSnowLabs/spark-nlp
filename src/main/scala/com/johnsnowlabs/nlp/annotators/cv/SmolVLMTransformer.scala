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
import com.johnsnowlabs.ml.ai.SmolVLM
import com.johnsnowlabs.ml.ai.SmolVLMConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.ml.util.Openvino
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import org.json4s.{DefaultFormats, JValue}
import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.SmolVLMWrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, BooleanParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** SmolVLMTransformer can load SmolVLM models for visual question answering. The model consists
  * of a vision encoder, a text encoder as well as a text decoder. The vision encoder will encode
  * the input image, the text encoder will encode the input question together with the encoding of
  * the image, and the text decoder will output the answer to the question.
  *
  * SmolVLM is a compact open multimodal model that accepts arbitrary sequences of image and text
  * inputs to produce text outputs. Designed for efficiency, SmolVLM can answer questions about
  * images, describe visual content, create stories grounded on multiple images, or function as a
  * pure language model without visual inputs. Its lightweight architecture makes it suitable for
  * on-device applications while maintaining strong performance on multimodal tasks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQA = SmolVLMTransformer.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"smolvlm_instruct"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/SmolVLMTransformerTest.scala]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF: DataFrame = ResourceHelper.spark.read
  *  .format("image")
  *  .option("dropInvalid", value = true)
  *  .load(imageFolder)
  *
  * val testDF: DataFrame = imageDF.withColumn("text", lit("<|im_start|>User:<image>Can you describe the image?<end_of_utterance>\nAssistant:"))
  *
  * val imageAssembler: ImageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQAClassifier = SmolVLMTransformer.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   imageAssembler,
  *   visualQAClassifier
  * ))
  *
  * val result = pipeline.fit(testDF).transform(testDF)
  *
  * result.select("image_assembler.origin", "answer.result").show(false)
  * +--------------------------------------+------+
  * |origin                                |result|
  * +--------------------------------------+------+
  * |[file:///content/images/cat_image.jpg]|[The unusual aspect of this picture is the presence of two cats lying on a pink couch]|
  * +--------------------------------------+------+
  * }}}
  *
  * @see
  *   [[CLIPForZeroShotClassification]] for Zero Shot Image Classifier
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based classifiers
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

class SmolVLMTransformer(override val uid: String)
    extends AnnotatorModel[SmolVLMTransformer]
    with HasBatchedAnnotateImage[SmolVLMTransformer]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("SmolVLMTransformer"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): SmolVLMTransformer.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): SmolVLMTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
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

  private var _model: Option[Broadcast[SmolVLM]] = None
  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  val imageToken =
    new IntParam(this, "imageToken", "Token id for image embeddings")

  /** @group setParam */
  def setImageToken(value: Int): this.type = set(imageToken, value)

  /** @group getParam */
  def getImageToken: Int = $(imageToken)

  val numVisionTokens =
    new IntParam(this, "numVisionTokens", "Number of vision tokens")

  /** @group setParam */
  def setNumVisionTokens(value: Int): this.type = set(numVisionTokens, value)

  /** @group getParam */
  def getNumVisionTokens: Int = $(numVisionTokens)

  val paddingConstant =
    new IntParam(this, "paddingConstant", "Padding constant for the model. Default is 0")

  /** @group setParam */
  def setPaddingConstant(value: Int): this.type = set(paddingConstant, value)

  /** @group getParam */
  def getPaddingConstant: Int = $(paddingConstant)

  val smolVLMConfig: StructFeature[SmolVLMConfig] =
    new StructFeature(this, "smolVLMConfig").setProtected()

  def setSmolVLMConfig(value: SmolVLMConfig): this.type =
    set(smolVLMConfig, value)

  def getSmolVLMConfig: SmolVLMConfig = $$(smolVLMConfig)

  val maxImageSize =
    new IntParam(this, "maxImageSize", "Maximum image size for the model. Default is 384")

  /** @group setParam */
  def setMaxImageSize(value: Int): this.type = set(maxImageSize, value)

  /** @group getParam */
  def getMaxImageSize: Int = $(maxImageSize)

  val doImageSplitting =
    new BooleanParam(this, "doImageSplitting", "Whether to split the image. Default is true")

  /** @group setParam */
  def setDoImageSplitting(value: Boolean): this.type = set(doImageSplitting, value)

  /** @group getParam */
  def getDoImageSplitting: Boolean = $(doImageSplitting)

  val patchSize =
    new IntParam(this, "patchSize", "Patch size for the model. Default is 14")

  def setPatchSize(value: Int): this.type = set(patchSize, value)

  def getPatchSize: Int = $(patchSize)

  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[SmolVLMWrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new SmolVLM(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor,
            generationConfig = getGenerationConfig,
            config = getSmolVLMConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: SmolVLM = _model.get.value

  setDefault(
    minOutputLength -> 0,
    maxOutputLength -> 20,
    doSample -> false,
    temperature -> 0.6,
    topK -> -1,
    topP -> 0.9,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 4096,
    stopTokenIds -> Array(49154),
    imageToken -> 49153,
    numVisionTokens -> 81,
    maxImageSize -> 384,
    patchSize -> 14,
    paddingConstant -> 0)

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations in batches that correspond to inputAnnotationCols generated by previous
    *   annotators if any
    * @return
    *   any number of annotations processed for every batch of input annotations. Not necessary
    *   one to one relationship
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationImage]]): Seq[Seq[Annotation]] = {

    batchedAnnotations
      //      .filter { annotationImages =>
      //        annotationImages.exists(_.text.nonEmpty)
      //      }
      .map { cleanAnnotationImages =>
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
          """<|im_start|>User:<image>Can you describe the image?<end_of_utterance>\nAssistant:""".stripMargin // default question
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
          Seq((wrappers.get.languageModel, "language_model.xml")),
          SmolVLMTransformer.suffix)
        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEncoderModel, "image_encoder.xml")),
          SmolVLMTransformer.suffix)
        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.modelMergerModel, "model_merger.xml")),
          SmolVLMTransformer.suffix)
        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbeddingsModel, "text_embeddings.xml")),
          SmolVLMTransformer.suffix)
        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.lmHeadModel, "lm_head.xml")),
          SmolVLMTransformer.suffix)
      case _ => throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedSmolVLMTransformer
    extends ParamsAndFeaturesReadable[SmolVLMTransformer]
    with HasPretrained[SmolVLMTransformer] {

  override val defaultModelName: Some[String] = Some("smolvlm_instruct")

  /** Java compliant-overrides */
  override def pretrained(): SmolVLMTransformer = super.pretrained()

  override def pretrained(name: String): SmolVLMTransformer =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): SmolVLMTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): SmolVLMTransformer =
    super.pretrained(name, lang, remoteLoc)

}

trait ReadSmolVLMTransformerDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[SmolVLMTransformer] =>
  val suffix: String = "_smolvlm"
  override val openvinoFile: String = "smolvlm_openvino"

  def readModel(instance: SmolVLMTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case Openvino.name =>
        val languageModelWrappers =
          readOpenvinoModels(path, spark, Seq("language_model.xml"), suffix)
        val visionEmbeddingsModelWrappers =
          readOpenvinoModels(path, spark, Seq("image_embed.xml"), suffix)
        val reshapeModelWrapper =
          readOpenvinoModels(path, spark, Seq("image_connector.xml"), suffix)
        val imageEncoderModelWrapper =
          readOpenvinoModels(path, spark, Seq("image_encoder.xml"), suffix)
        val modelMergerModelWrapper =
          readOpenvinoModels(path, spark, Seq("model_merger.xml"), suffix)
        val textEmbeddingsModelWrapper =
          readOpenvinoModels(path, spark, Seq("text_embeddings.xml"), suffix)
        val lmHeadModelWrapper = readOpenvinoModels(path, spark, Seq("lm_head.xml"), suffix)

        val ovWrapper = SmolVLMWrappers(
          languageModel = languageModelWrappers("language_model.xml"),
          imageEmbedModel = visionEmbeddingsModelWrappers("image_embed.xml"),
          imageEncoderModel = imageEncoderModelWrapper("image_encoder.xml"),
          imageConnectorModel = reshapeModelWrapper("image_connector.xml"),
          modelMergerModel = modelMergerModelWrapper("model_merger.xml"),
          textEmbeddingsModel = textEmbeddingsModelWrapper("text_embeddings.xml"),
          lmHeadModel = lmHeadModelWrapper("lm_head.xml"))

        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "SmolVLMFeatureExtractor",
          instance.getImageMean,
          instance.getImageStd,
          instance.getResample,
          instance.getSize)

        instance.setModelIfNotSet(spark, preprocessor, None, Some(ovWrapper))
      case _ => throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): SmolVLMTransformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom = Some(
          List(
            "language_model",
            "image_embed",
            "image_connector",
            "image_encoder",
            "model_merger",
            "text_embeddings",
            "lm_head")))

//    config.json
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

//    preprocessor_config.json
    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig = Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)
    val parsedPreprocessorConfig: JValue = parse(preprocessorConfigJsonContent)

    //   Values from config.json
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

    val vocabSize = (modelConfig \ "vocab_size").extract[Int]
    val imageToken = (modelConfig \ "image_token_id").extract[Int]
    val imageSeqLen = (modelConfig \ "image_seq_len").extract[Int]
    val patchSize = (modelConfig \ "vision_config" \ "patch_size").extract[Int]

//    Values from preprocessor_config.json that are not in the preprocessor class
    val doImageSplitting = (parsedPreprocessorConfig \ "do_image_splitting").extract[Boolean]
    val maxImageSize = (parsedPreprocessorConfig \ "max_image_size").extract[Map[String, Int]]

    val tokenizerConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "tokenizer_config.json"))
    val bosToken = (tokenizerConfig \ "bos_token").extract[String]
    val eosToken = (tokenizerConfig \ "eos_token").extract[String]
    val padToken = (tokenizerConfig \ "pad_token").extract[String]
    val unk_token = (tokenizerConfig \ "unk_token").extract[String]

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

    val bosTokenId = vocabs(bosToken)
    val eosTokenId = vocabs(eosToken)
    val padTokenId = vocabs(padToken)
    val unkTokenId = vocabs(unk_token)

    val annotatorModel = new SmolVLMTransformer()
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
      .setImageToken(imageToken)
      .setSize(preprocessorConfig.size)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setNumVisionTokens(imageSeqLen)
      .setDoImageSplitting(doImageSplitting)
      .setMaxImageSize(maxImageSize("longest_edge"))
      .setSmolVLMConfig(SmolVLMConfig(
        doResize = preprocessorConfig.do_resize,
        size = Map("longest_edge" -> preprocessorConfig.size),
        maxImageSize = Map("longest_edge" -> maxImageSize("longest_edge")),
        doRescale = preprocessorConfig.do_rescale,
        rescaleFactor = preprocessorConfig.rescale_factor,
        doNormalize = preprocessorConfig.do_normalize,
        resample = preprocessorConfig.resample,
        imageMean = preprocessorConfig.image_mean,
        imageStd = preprocessorConfig.image_std,
        doImageSplitting = doImageSplitting,
        imageTokenId = imageToken,
        imageSeqLen = imageSeqLen,
        unkTokenId = unkTokenId,
        patchSize = patchSize))

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    detectedEngine match {
      case Openvino.name =>
        val languageModelWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "language_model")
        val imageEmbedWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "image_embed")
        val imageEncoderWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "image_encoder")
        val imageConnectorWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "image_connector")
        val modelMergerWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "model_merger")
        val textEmbeddingsWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "text_embeddings")
        val lmHeadWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "lm_head")

        val openvinoWrapper = SmolVLMWrappers(
          languageModel = languageModelWrapper,
          imageEmbedModel = imageEmbedWrapper,
          imageEncoderModel = imageEncoderWrapper,
          imageConnectorModel = imageConnectorWrapper,
          modelMergerModel = modelMergerWrapper,
          textEmbeddingsModel = textEmbeddingsWrapper,
          lmHeadModel = lmHeadWrapper)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object SmolVLMTransformer
    extends ReadablePretrainedSmolVLMTransformer
    with ReadSmolVLMTransformerDLModel
