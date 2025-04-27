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
import com.johnsnowlabs.ml.ai.Gemma3
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
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Gemma3Wrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Gemma3ForMultiModal can load Gemma3 Vision models for visual question answering. The model
  * consists of a vision encoder, a text encoder, a text decoder and a model merger. The vision
  * encoder will encode the input image, the text encoder will encode the input text, the model
  * merger will merge the image and text embeddings, and the text decoder will output the answer.
  *
  * Gemma 3 is a family of lightweight, state-of-the-art open models from Google, built from the
  * same research and technology used to create the Gemini models. Key features include:
  *   - Large 128K context window
  *   - Multilingual support in over 140 languages
  *   - Multimodal capabilities handling both text and image inputs
  *   - Optimized for deployment on limited resources (laptops, desktops, cloud)
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQA = Gemma3ForMultiModal.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"gemma3_4b_it_int4"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF = spark.read
  *   .format("image")
  *   .option("dropInvalid", value = true)
  *   .load(imageFolder)
  *
  * val testDF = imageDF.withColumn("text", lit("<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n"))
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQA = Gemma3ForMultiModal.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   imageAssembler,
  *   visualQA
  * ))
  *
  * val result = pipeline.fit(testDF).transform(testDF)
  *
  * result.select("image_assembler.origin", "answer.result").show(truncate = false)
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  */

class Gemma3ForMultiModal(override val uid: String)
    extends AnnotatorModel[Gemma3ForMultiModal]
    with HasBatchedAnnotateImage[Gemma3ForMultiModal]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("Gemma3ForMultiModal"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): Gemma3ForMultiModal.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): Gemma3ForMultiModal.this.type = {
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

  private var _model: Option[Broadcast[Gemma3]] = None
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

  val imageTokenLength =
    new IntParam(this, "imageTokenLength", "Token length for image embeddings")

  /** @group setParam */
  def setImageTokenLength(value: Int): this.type = set(imageTokenLength, value)

  /** @group getParam */
  def getImageTokenLength: Int = $(imageTokenLength)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[Gemma3Wrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Gemma3(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor,
            generationConfig = getGenerationConfig,
            imageToken = getImageToken,
            imageTokenLength = getImageTokenLength)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Gemma3 = _model.get.value

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
    stopTokenIds -> Array(2),
    imageToken -> 257152,
    imageTokenLength -> 256)

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
          "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n<start_of_image>Describe this image in detail.<end_of_turn>\n<start_of_turn>model\n" // default question
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
          Seq((wrappers.get.languageModel, "openvino_language_model.xml")),
          Gemma3ForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEncoder, "openvino_vision_embeddings_model.xml")),
          Gemma3ForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbeddings, "openvino_text_embeddings_model.xml")),
          Gemma3ForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.modelMerger, "openvino_merger_model.xml")),
          Gemma3ForMultiModal.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }
}

trait ReadablePretrainedGemma3ForMultiModal
    extends ParamsAndFeaturesReadable[Gemma3ForMultiModal]
    with HasPretrained[Gemma3ForMultiModal] {

  override val defaultModelName: Some[String] = Some("gemma3_4b_it_int4")

  /** Java compliant-overrides */
  override def pretrained(): Gemma3ForMultiModal = super.pretrained()

  override def pretrained(name: String): Gemma3ForMultiModal =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): Gemma3ForMultiModal =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): Gemma3ForMultiModal =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadGemma3ForMultiModalDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[Gemma3ForMultiModal] =>
  val suffix: String = "_Gemma3"
  override val openvinoFile: String = "Gemma3_openvino"
  def readModel(instance: Gemma3ForMultiModal, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case Openvino.name =>
        val languageModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_language_model.xml"), suffix)

        val imageEncoderWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_vision_embeddings_model.xml"), suffix)

        val textEmbeddingsWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_text_embeddings_model.xml"), suffix)

        val modelMergerWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_merger_model.xml"), suffix)

        val ovWrapper = Gemma3Wrappers(
          languageModel = languageModelWrappers("openvino_language_model.xml"),
          imageEncoder = imageEncoderWrappers("openvino_vision_embeddings_model.xml"),
          textEmbeddings = textEmbeddingsWrappers("openvino_text_embeddings_model.xml"),
          modelMerger = modelMergerWrappers("openvino_merger_model.xml"))
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "Gemma3FeatureExtractor",
          instance.getImageMean,
          instance.getImageStd,
          instance.getResample,
          instance.getSize)
        instance.setModelIfNotSet(spark, preprocessor, None, Some(ovWrapper))
      case _ => {
        throw new Exception(notSupportedEngineError)
      }
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): Gemma3ForMultiModal = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s
    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom = Some(
          List(
            "openvino_language_model",
            "openvino_vision_embeddings_model",
            "openvino_text_embeddings_model",
            "openvino_merger_model")))
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val generationConfigJson: JValue = parse(
      loadJsonStringAsset(localModelPath, "generation_config.json"))

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

    val bosTokenId = (generationConfigJson \ "bos_token_id").extract[Int]
    val eosTokenIdArray = (generationConfigJson \ "eos_token_id").extract[Array[Int]]
    val eosTokenId = eosTokenIdArray.head
//    val eosTokenId = (generationConfigJson \ "eos_token_id").extract[Int]
    val padTokenId = (generationConfigJson \ "pad_token_id").extract[Int]
    val vocabSize = (modelConfig \ "text_config" \ "vocab_size").extract[Int]

    val imageToken = (modelConfig \ "image_token_index").extract[Int]
    val imageTokenLength = (modelConfig \ "mm_tokens_per_image").extract[Int]

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

    val annotatorModel = new Gemma3ForMultiModal()
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
      .setImageTokenLength(imageTokenLength)
      .setSize(preprocessorConfig.size)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setStopTokenIds(eosTokenIdArray)

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    detectedEngine match {
      case Openvino.name =>
        val imageEncoderWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_vision_embeddings_model")
        val textEmbeddingsWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_text_embeddings_model")
        val modelMergerWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_merger_model")
        val languageModelWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_language_model")

        val openvinoWrapper = Gemma3Wrappers(
          languageModel = languageModelWrapper,
          imageEncoder = imageEncoderWrapper,
          textEmbeddings = textEmbeddingsWrapper,
          modelMerger = modelMergerWrapper)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object Gemma3ForMultiModal
    extends ReadablePretrainedGemma3ForMultiModal
    with ReadGemma3ForMultiModalDLModel
