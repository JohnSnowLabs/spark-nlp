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
import com.johnsnowlabs.ml.ai.InternVL
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
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.InternVLWrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** InternVLForMultiModal can load InternVL Vision models for visual question answering. The model
  * consists of a vision encoder, a text encoder, a text decoder and a model merger. The vision
  * encoder will encode the input image, the text encoder will encode the input text, the model
  * merger will merge the image and text embeddings, and the text decoder will output the answer.
  *
  * InternVL 2.5 is an advanced multimodal large language model (MLLM) series that builds upon
  * InternVL 2.0, maintaining its core model architecture while introducing significant
  * enhancements in training and testing strategies as well as data quality. Key features include:
  *   - Large context window support
  *   - Multilingual support
  *   - Multimodal capabilities handling both text and image inputs
  *   - Optimized for deployment with int4 quantization
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQA = InternVLForMultiModal.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"internvl2_5_1b_int4"`, if no name is provided.
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
  * val testDF = imageDF.withColumn("text", lit("<|im_start|><image>\nDescribe this image in detail.<|im_end|><|im_start|>assistant\n"))
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQA = InternVLForMultiModal.pretrained()
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

class InternVLForMultiModal(override val uid: String)
    extends AnnotatorModel[InternVLForMultiModal]
    with HasBatchedAnnotateImage[InternVLForMultiModal]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("InternVLForMultiModal"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): InternVLForMultiModal.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): InternVLForMultiModal.this.type = {
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

  private var _model: Option[Broadcast[InternVL]] = None
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
      openvinoWrapper: Option[InternVLWrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new InternVL(
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
  def getModelIfNotSet: InternVL = _model.get.value

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
          "<|im_start|><image>\nDescribe this image in detail.<|im_end|><|im_start|>assistant\n" // default question
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
          InternVLForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEncoder, "openvino_vision_embeddings_model.xml")),
          InternVLForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbeddings, "openvino_text_embeddings_model.xml")),
          InternVLForMultiModal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.modelMerger, "openvino_merger_model.xml")),
          InternVLForMultiModal.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }
}

trait ReadablePretrainedInternVLForMultiModal
    extends ParamsAndFeaturesReadable[InternVLForMultiModal]
    with HasPretrained[InternVLForMultiModal] {

  override val defaultModelName: Some[String] = Some("internvl2_5_1b_int4")

  /** Java compliant-overrides */
  override def pretrained(): InternVLForMultiModal = super.pretrained()

  override def pretrained(name: String): InternVLForMultiModal =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): InternVLForMultiModal =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): InternVLForMultiModal =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadInternVLForMultiModalDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[InternVLForMultiModal] =>
  val suffix: String = "_InternVL"
  override val openvinoFile: String = "InternVL_openvino"
  def readModel(instance: InternVLForMultiModal, path: String, spark: SparkSession): Unit = {
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

        val ovWrapper = InternVLWrappers(
          languageModel = languageModelWrappers("openvino_language_model.xml"),
          imageEncoder = imageEncoderWrappers("openvino_vision_embeddings_model.xml"),
          textEmbeddings = textEmbeddingsWrappers("openvino_text_embeddings_model.xml"),
          modelMerger = modelMergerWrappers("openvino_merger_model.xml"))
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "InternVLFeatureExtractor",
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
      useOpenvino: Boolean = false): InternVLForMultiModal = {
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

    val bosTokenId = (modelConfig \ "llm_config" \ "bos_token_id").extract[Int]
    var eosTokenIdArray: Array[Int] = Array()
    var eosTokenId: Int = 0
    try {
      eosTokenIdArray = (generationConfigJson \ "eos_token_id").extract[Array[Int]]
      eosTokenId = eosTokenIdArray.head
    } catch {
      case _: Exception =>
        eosTokenId = (modelConfig \ "llm_config" \ "eos_token_id").extract[Int]
        eosTokenIdArray = Array(eosTokenId)
    }

    //    val eosTokenId = (generationConfigJson \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "llm_config" \ "bos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "llm_config" \ "vocab_size").extract[Int]

    val imageToken = (modelConfig \ "img_context_token_id").extract[Int]
    var imageSize: Int = 448
    try {
      imageSize = (modelConfig \ "force_image_size").extract[Int]
    } catch {
      case _: Exception =>
        imageSize = (modelConfig \ "vision_config" \ "image_size").extract[Int]
    }
    val patchSize =
      (modelConfig \ "vision_config" \ "patch_size").extract[Int]

    val downsampleRatio =
      (modelConfig \ "downsample_ratio").extract[Float]
    val imageTokenLength =
      ((imageSize / patchSize).toInt * (imageSize / patchSize).toInt * (downsampleRatio * downsampleRatio)).toInt
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

    val annotatorModel = new InternVLForMultiModal()
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

        val openvinoWrapper = InternVLWrappers(
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

object InternVLForMultiModal
    extends ReadablePretrainedInternVLForMultiModal
    with ReadInternVLForMultiModalDLModel
