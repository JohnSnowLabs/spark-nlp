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
import com.johnsnowlabs.ml.ai.MLLama
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
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.MLLamaWrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** MLLamaForMultimodal can load LLAMA 3.2 Vision models for visual question answering. The model
  * consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder
  * will encode the input image, the text encoder will encode the input question together with the
  * encoding of the image, and the text decoder will output the answer to the question.
  *
  * The Llama 3.2-Vision collection of multimodal large language models (LLMs) is a collection of
  * pretrained and instruction-tuned image reasoning generative models in 11B and 90B sizes (text
  * + images in / text out). The Llama 3.2-Vision instruction-tuned models are optimized for
  * visual recognition, image reasoning, captioning, and answering general questions about an
  * image. The models outperform many of the available open source and closed multimodal models on
  * common industry benchmarks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQA = MLLamaForMultimodal.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"llama_3_2_11b_vision_instruct_int4"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/MLLamaForMultimodalTest.scala]].
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
  * val testDF: DataFrame = imageDF.withColumn("text", lit("<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|image|>What is unusual on this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"))
  *
  * val imageAssembler: ImageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQAClassifier = MLLamaForMultimodal.pretrained()
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

class MLLamaForMultimodal(override val uid: String)
    extends AnnotatorModel[MLLamaForMultimodal]
    with HasBatchedAnnotateImage[MLLamaForMultimodal]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("MLLamaForMultimodal"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): MLLamaForMultimodal.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): MLLamaForMultimodal.this.type = {
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

  private var _model: Option[Broadcast[MLLama]] = None
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

  val maxImageTiles =
    new IntParam(this, "maxImageTiles", "Maximum number of image tiles")

  /** @group setParam */
  def setMaxImageTiles(value: Int): this.type = set(maxImageTiles, value)

  /** @group getParam */
  def getMaxImageTiles: Int = $(maxImageTiles)

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

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[MLLamaWrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MLLama(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor,
            generationConfig = getGenerationConfig,
            imageToken = getImageToken,
            maxImageTiles = getMaxImageTiles,
            numVisionTokens = getNumVisionTokens,
            paddingConstant = getPaddingConstant)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: MLLama = _model.get.value

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
    stopTokenIds -> Array(128001, 128008, 128009),
    imageToken -> 128256,
    maxImageTiles -> 576,
    numVisionTokens -> 1601,
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
          """<|begin_of_text|><|begin_of_text|><|start_header_id|>user<|end_header_id|>\n""" +
            """\n<|image|>This is an image<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n""".stripMargin // default question
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
          Seq(
            (
              wrappers.get.languageModel,
              "llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers.xml")),
          MLLamaForMultimodal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.visionEmbeddingsModel, "openvino_vision_encoder.xml")),
          MLLamaForMultimodal.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.reshapeModel, "openvino_reshape_model.xml")),
          MLLamaForMultimodal.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedMLLamaForMultimodal
    extends ParamsAndFeaturesReadable[MLLamaForMultimodal]
    with HasPretrained[MLLamaForMultimodal] {

  override val defaultModelName: Some[String] = Some("llama_3_2_11b_vision_instruct_int4")

  /** Java compliant-overrides */
  override def pretrained(): MLLamaForMultimodal = super.pretrained()

  override def pretrained(name: String): MLLamaForMultimodal =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): MLLamaForMultimodal =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MLLamaForMultimodal =
    super.pretrained(name, lang, remoteLoc)

}

trait ReadMLLamaForMultimodalDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[MLLamaForMultimodal] =>
  val suffix: String = "_mllama"
  override val openvinoFile: String = "mllama_openvino"
  def readModel(instance: MLLamaForMultimodal, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case Openvino.name =>
        val languageModelWrappers =
          readOpenvinoModels(
            path,
            spark,
            Seq("llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers.xml"),
            suffix)

        val visionEmbeddingsModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_vision_encoder.xml"), suffix)

        val reshapeModelWrapper =
          readOpenvinoModels(path, spark, Seq("openvino_reshape_model.xml"), suffix)

        val ovWrapper = MLLamaWrappers(
          languageModel = languageModelWrappers(
            "llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers.xml"),
          visionEmbeddingsModel = visionEmbeddingsModelWrappers("openvino_vision_encoder.xml"),
          reshapeModel = reshapeModelWrapper("openvino_reshape_model.xml"))
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "LLAVAFeatureExtractor",
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
      useOpenvino: Boolean = false): MLLamaForMultimodal = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom = Some(
          List(
            "llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers",
            "openvino_vision_encoder",
            "openvino_reshape_model")))
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))
    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig = Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    val parsedPreprocessorConfig: JValue = parse(preprocessorConfigJsonContent)
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

    val maxImageTiles = (parsedPreprocessorConfig \ "max_image_tiles").extract[Int]

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val generationConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "generation_config.json"))
    val bosTokenId = (generationConfig \ "bos_token_id").extract[Int]
    val eosTokenIdArray = (generationConfig \ "eos_token_id").extract[Array[Int]]
    val eosTokenId = eosTokenIdArray.head
    val padTokenId = (generationConfig \ "pad_token_id").extract[Int]
    val vocabSize = (modelConfig \ "text_config" \ "vocab_size").extract[Int]

    val imageToken = (modelConfig \ "image_token_index").extract[Int]
    val imageSize = (modelConfig \ "vision_config" \ "image_size").extract[Int]
    val patchSize = (modelConfig \ "vision_config" \ "patch_size").extract[Int]

    val numVisionTokens = Math
      .pow(imageSize / patchSize, 2)
      .toInt + 1

//    val numVisionTokens = Math
//      .pow(
//        ((modelConfig \ "vision_config" \ "image_size")
//          .extract[Int] / (modelConfig \ "vision_config" \ "patch_size").extract[Int]).toInt,
//        2)
//      .toInt + 1

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
//    val vocabSize = vocabs.size
    val annotatorModel = new MLLamaForMultimodal()
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
      .setMaxImageTiles(maxImageTiles)
      .setNumVisionTokens(numVisionTokens)
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
        val visionWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_vision_encoder")
        val reshapeWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_reshape_model")
        val languageModelWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "llm_int4_asym_r10_gs64_max_activation_variance_awq_scale_all_layers")

        val openvinoWrapper = MLLamaWrappers(
          languageModel = languageModelWrapper,
          visionEmbeddingsModel = visionWrapper,
          reshapeModel = reshapeWrapper)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object MLLamaForMultimodal
    extends ReadablePretrainedMLLamaForMultimodal
    with ReadMLLamaForMultimodalDLModel
