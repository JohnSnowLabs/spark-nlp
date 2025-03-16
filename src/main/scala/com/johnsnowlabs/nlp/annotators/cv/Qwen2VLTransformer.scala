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
import com.johnsnowlabs.ml.ai.Qwen2VL
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.Openvino
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import org.json4s.{DefaultFormats, JValue}
import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.Qwen2VLWrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Qwen2VLTransformer can load Qwen2 Vision-Language models for visual question answering and
  * multimodal instruction following. The model consists of a vision encoder, a text encoder, and
  * a text decoder. The vision encoder processes the input image, the text encoder integrates the
  * encoding of the image with the input text, and the text decoder outputs the response to the
  * query or instruction.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQA = Qwen2VLTransformer.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"qwen2_vl_2b_instruct_int4"`, if no name is provided.
  *
  * For available pretrained models, please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]]. To explore more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/Qwen2VLTransformerTest.scala]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF: DataFrame = ResourceHelper.spark.read
  *   .format("image")
  *   .option("dropInvalid", value = true)
  *   .load(imageFolder)
  *
  * val testDF: DataFrame = imageDF.withColumn("text", lit("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"))
  *
  * val imageAssembler: ImageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQAClassifier = Qwen2VLTransformer.pretrained()
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
  * |[file:///content/images/cat_image.jpg]|[This image is unusual because it features two cats lying on a pink couch.]|
  * +--------------------------------------+------+
  * }}}
  *
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer-
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
class Qwen2VLTransformer(override val uid: String)
    extends AnnotatorModel[Qwen2VLTransformer]
    with HasBatchedAnnotateImage[Qwen2VLTransformer]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("Qwen2VLTransformer"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): Qwen2VLTransformer.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): Qwen2VLTransformer.this.type = {
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

  /** max pixel values for image normalization
    *
    * @group param
    */
  val maxPixelValue =
    new IntParam(this, "maxPixelValue", "max pixel values for image normalization")

  /** @group setParam */
  def setMaxPixelValue(value: Int): this.type = {
    set(maxPixelValue, value)
  }

  /** @group getParam */
  def getMaxPixelValue: Int = $(maxPixelValue)

  /** min pixel values for image normalization
    *
    * @group param
    */
  val minPixelValue =
    new IntParam(this, "minPixelValue", "min pixel values for image normalization")

  /** @group setParam */
  def setMinPixelValue(value: Int): this.type = {
    set(minPixelValue, value)
  }

  /** @group getParam */
  def getMinPixelValue: Int = $(minPixelValue)

  private var _model: Option[Broadcast[Qwen2VL]] = None
  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[Qwen2VLWrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Qwen2VL(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor = preprocessor,
            generationConfig = getGenerationConfig,
            minPixels = $(minPixelValue),
            maxPixels = $(maxPixelValue))))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: Qwen2VL = _model.get.value

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
    stopTokenIds -> Array(128001),
    maxPixelValue -> 16384 * 28 * 28,
    minPixelValue -> 256 * 28 * 28)

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
          "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n" // default question
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
          Seq((wrappers.get.patchReshapeModel, "openvino_patch_reshape_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.languageModel, "openvino_language_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEmbedding, "openvino_vision_embeddings_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imageEmbeddingMerger, "openvino_vision_embeddings_merger_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbedding, "openvino_text_embeddings_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.multimodalMergeModel, "openvino_multimodal_merge_model.xml")),
          Qwen2VLTransformer.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.rotaryEmbedding, "openvino_rotary_embeddings_model.xml")),
          Qwen2VLTransformer.suffix)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedQwen2VLTransformer
    extends ParamsAndFeaturesReadable[Qwen2VLTransformer]
    with HasPretrained[Qwen2VLTransformer] {

  override val defaultModelName: Some[String] = Some("qwen2_vl_2b_instruct_int4")

  /** Java compliant-overrides */
  override def pretrained(): Qwen2VLTransformer = super.pretrained()

  override def pretrained(name: String): Qwen2VLTransformer =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): Qwen2VLTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): Qwen2VLTransformer =
    super.pretrained(name, lang, remoteLoc)

}

trait ReadQwen2VLTransformerDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[Qwen2VLTransformer] =>
  val suffix: String = "_qwen2vl"
  override val openvinoFile: String = "qwen2vl_openvino"
  def readModel(instance: Qwen2VLTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      // LANGUAGE_MODEL_NAME = "openvino_language_model.xml"
      // IMAGE_EMBEDDING_NAME = "openvino_vision_embeddings_model.xml"
      // IMAGE_EMBEDDING_MERGER_NAME = "openvino_vision_embeddings_merger_model.xml"
      // TEXT_EMBEDDING_NAME = "openvino_text_embeddings_model.xml"
      // ROTARY_EMBEDDING_NAME = "openvino_rotary_embeddings_model.xml"
      // PATCH_RESHAPE_NAME = "openvino_patch_reshape_model.xml"
      case Openvino.name =>
        val languageModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_language_model.xml"), suffix)
        val imageEmbeddingWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_vision_embeddings_model.xml"), suffix)
        val imageEmbeddingMergerWrappers =
          readOpenvinoModels(
            path,
            spark,
            Seq("openvino_vision_embeddings_merger_model.xml"),
            suffix)
        val textEmbeddingWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_text_embeddings_model.xml"), suffix)
        val rotaryEmbeddingWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_rotary_embeddings_model.xml"), suffix)
        val patchReshapeWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_patch_reshape_model.xml"), suffix)
        val multiModalMergeWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_multimodal_merge_model.xml"), suffix)
        val ovWrapper = Qwen2VLWrappers(
          imageEmbedding = imageEmbeddingWrappers("openvino_vision_embeddings_model.xml"),
          imageEmbeddingMerger =
            imageEmbeddingMergerWrappers("openvino_vision_embeddings_merger_model.xml"),
          languageModel = languageModelWrappers("openvino_language_model.xml"),
          textEmbedding = textEmbeddingWrappers("openvino_text_embeddings_model.xml"),
          rotaryEmbedding = rotaryEmbeddingWrappers("openvino_rotary_embeddings_model.xml"),
          patchReshapeModel = patchReshapeWrappers("openvino_patch_reshape_model.xml"),
          multimodalMergeModel = multiModalMergeWrappers("openvino_multimodal_merge_model.xml"))
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "Qwen2VLFeatureExtractor",
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
      useOpenvino: Boolean = false): Qwen2VLTransformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom = Some(
          List(
            "openvino_text_embeddings_model",
            "openvino_language_model",
            "openvino_vision_embeddings_model",
            "openvino_vision_embeddings_merger_model",
            "openvino_rotary_embeddings_model",
            "openvino_patch_reshape_model",
            "openvino_multimodal_merge_model")))
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
    val padTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    // Check if tokenizer.json exists
    val tokenizerPath = s"$localModelPath/assets/tokenizer.json"
    val tokenizerExists = new java.io.File(tokenizerPath).exists()
    val (vocabs, addedTokens, bytePairs) = if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      // extract vocab from tokenizer.json ( model -> vocab)
      var vocabs: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]

      // extract merges from tokenizer.json ( model -> merges)
      val bytePairs = (tokenizerConfig \ "model" \ "merges")
        .extract[List[Array[String]]]
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap

      // extract added_tokens from tokenizer.json (added_tokens)
      // "added_tokens": [
      //    {
      //      "id": 128000,
      //      "content": "<|begin_of_text|>",
      //      "single_word": false,
      //      "lstrip": false,
      //      "rstrip": false,
      //      "normalized": false,
      //      "special": true
      //    }, ...
      //  ]
      val addedTokens = (tokenizerConfig \ "added_tokens")
        .extract[List[Map[String, Any]]]
        .map { token =>
          val id = token("id").asInstanceOf[BigInt].intValue()
          val content = token("content").asInstanceOf[String]
          (content, id)
        }
        .toMap

      // update vocab with added tokens
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

    val annotatorModel = new Qwen2VLTransformer()
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
        val patchReshapeWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_patch_reshape_model")

        val languageModelWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_language_model")

        val imageEmbeddingWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_vision_embeddings_model")

        val imageEmbeddingMergerWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_vision_embeddings_merger_model")

        val textEmbeddingWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_text_embeddings_model")

        val rotaryEmbeddingWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_rotary_embeddings_model")

        val multimodalMergerWrappers =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_multimodal_merge_model")

        val openvinoWrapper = Qwen2VLWrappers(
          languageModel = languageModelWrappers,
          imageEmbedding = imageEmbeddingWrappers,
          imageEmbeddingMerger = imageEmbeddingMergerWrappers,
          textEmbedding = textEmbeddingWrappers,
          rotaryEmbedding = rotaryEmbeddingWrappers,
          patchReshapeModel = patchReshapeWrappers,
          multimodalMergeModel = multimodalMergerWrappers)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object Qwen2VLTransformer
    extends ReadablePretrainedQwen2VLTransformer
    with ReadQwen2VLTransformerDLModel
