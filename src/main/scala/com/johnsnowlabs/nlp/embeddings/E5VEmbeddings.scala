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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.E5V
import com.johnsnowlabs.ml.onnx.OnnxWrapper.DecoderWrappers
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.ml.util.Openvino
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp._
import org.json4s.{DefaultFormats, JValue}
import org.json4s.jackson.JsonMethods.parse
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.openvino.OpenvinoWrapper.E5VWrappers
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{BinaryType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{Row, SparkSession}

/** E5VEmbeddings provides universal multimodal embeddings using the E5-V model, which is
  * fine-tuned from lmms-lab/llama3-llava-next-8b.
  *
  * E5-V bridges the modality gap between different input types (text, image) and demonstrates
  * strong performance in multimodal embeddings, even without fine-tuning. It also supports a
  * single-modality training approach, where the model is trained exclusively on text pairs, often
  * yielding better performance than multimodal training.
  *
  * For more details, see the Hugging Face model card: https://huggingface.co/royokong/e5-v
  *
  * ==Overview==
  *
  * E5-V can embed both text and images into a shared space, enabling cross-modal retrieval and
  * similarity tasks. The model is designed for universal embeddings and is suitable for scenarios
  * where you want to compare or retrieve across modalities.
  *
  * ==Example==
  *
  * ===Image + Text Embedding===
  * {{ { import org.apache.spark.sql.functions.lit import com.johnsnowlabs.nlp.base.ImageAssembler
  * import com.johnsnowlabs.nlp.embeddings.E5VEmbeddings import org.apache.spark.ml.Pipeline
  *
  * val imageDF = spark.read.format("image").option("dropInvalid", value = true).load(imageFolder)
  * val imagePrompt = "<|start_header_id|>user<|end_header_id|>\n\n<image>\\nSummary above image
  * in one word: <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n" val testDF =
  * imageDF.withColumn("text", lit(imagePrompt))
  *
  * val imageAssembler = new ImageAssembler().setInputCol("image").setOutputCol("image_assembler")
  * val e5vEmbeddings = E5VEmbeddings.pretrained() .setInputCols("image_assembler")
  * .setOutputCol("e5v")
  *
  * val pipeline = new Pipeline().setStages(Array(imageAssembler, e5vEmbeddings)) val result =
  * pipeline.fit(testDF).transform(testDF) result.select("e5v.embeddings").show(truncate = false)
  * }}
  *
  * ===Text-Only Embedding===
  * {{ { import org.apache.spark.sql.SparkSession import org.apache.spark.sql.functions.lit import
  * com.johnsnowlabs.nlp.util.EmbeddingsDataFrameUtils.{emptyImageRow, imageSchema} import
  * com.johnsnowlabs.nlp.embeddings.E5VEmbeddings
  *
  * val spark: SparkSession = ... val textPrompt =
  * "<|start_header_id|>user<|end_header_id|>\n\n<sent>\\nSummary above sentence in one word:
  * <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n" val textDesc = "A cat sitting
  * in a box." val nullImageDF =
  * spark.createDataFrame(spark.sparkContext.parallelize(Seq(emptyImageRow)), imageSchema) val
  * textDF = nullImageDF.withColumn("text", lit(textPrompt.replace("<sent>", textDesc)))
  *
  * val e5vEmbeddings = E5VEmbeddings.pretrained() .setInputCols("image") .setOutputCol("e5v") val
  * result = e5vEmbeddings.transform(textDF) result.select("e5v.embeddings").show(truncate =
  * false) }}
  *
  * ==References==
  *   - Hugging Face model card: https://huggingface.co/royokong/e5-v
  *   - Paper: https://arxiv.org/abs/2407.12580
  *   - Code: https://github.com/kongds/E5-V
  *
  * @see
  *   [[CLIPForZeroShotClassification]] for Zero Shot Image Classifier
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based classifiers
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

class E5VEmbeddings(override val uid: String)
    extends AnnotatorModel[E5VEmbeddings]
    with HasBatchedAnnotateImage[E5VEmbeddings]
    with HasImageFeatureProperties
    with WriteOpenvinoModel
    with HasGeneratorProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("E5VEmbeddings"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = SENTENCE_EMBEDDINGS

  /** @group setParam */
  def setRandomSeed(value: Int): E5VEmbeddings.this.type = {
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
  def setIgnoreTokenIds(tokenIds: Array[Int]): E5VEmbeddings.this.type = {
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

  private var _model: Option[Broadcast[E5V]] = None
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

  /** Pinpoints for image grid, used to extract image features from the grid
    *
    * @group param
    */
  val imageGridPinpoints: MapFeature[Int, Array[Int]] = new MapFeature(this, "imageGridPinpoints")

  /** @group setParam */
  def setImageGridPinpoints(value: Map[Int, Array[Int]]): this.type =
    set(imageGridPinpoints, value)

  /** @group getParam */
  def getImageGridPinpoints: Map[Int, Array[Int]] = $$(imageGridPinpoints)

  /** Patch size for image embeddings
    *
    * @group param
    */
  val patchSize: IntParam =
    new IntParam(this, "patchSize", "Patch size for image embeddings, default is 336")

  /** @group setParam */
  def setPatchSize(value: Int): this.type = set(patchSize, value)

  /** @group getParam */
  def getPatchSize: Int = $(patchSize)

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      onnxWrappers: Option[DecoderWrappers],
      openvinoWrapper: Option[E5VWrappers]): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new E5V(
            onnxWrappers,
            openvinoWrapper,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            preprocessor,
            generationConfig = getGenerationConfig,
            imageToken = getImageToken,
            imageGridPinpoints = getImageGridPinpoints,
            patchSize = getPatchSize)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: E5V = _model.get.value

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
    imageToken -> 128256,
    patchSize -> 336)

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
        val validImages = cleanAnnotationImages
        val questionAnnotations = extractInputAnnotation(validImages)

        getModelIfNotSet.predict(questionAnnotations, validImages.toSeq)
      }
  }

  private def extractInputAnnotation(
      annotationImages: Array[AnnotationImage]): Seq[Annotation] = {
    val questions = annotationImages.map(annotationImage => {
      val imageText =
        if (annotationImage.text.nonEmpty) annotationImage.text
        else
          "<|user|> \n <|image|> This is an image\n <|end|>\n <|assistant|>\n" // default question
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
          Seq((wrappers.get.languageModel, "openvino_language_model-int4.xml")),
          E5VEmbeddings.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.visionEmbeddingsModel, "openvino_vision_embeddings_model.xml")),
          E5VEmbeddings.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.textEmbeddingsModel, "openvino_text_embeddings_model.xml")),
          E5VEmbeddings.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.imagePackerModel, "openvino_image_packer.xml")),
          E5VEmbeddings.suffix)

        writeOpenvinoModels(
          path,
          spark,
          Seq((wrappers.get.mergeModel, "openvino_multimodal_merger.xml")),
          E5VEmbeddings.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedE5VEmbeddings
    extends ParamsAndFeaturesReadable[E5VEmbeddings]
    with HasPretrained[E5VEmbeddings] {

  override val defaultModelName: Some[String] = Some("e5v_1_5_7b_int4")

  /** Java compliant-overrides */
  override def pretrained(): E5VEmbeddings = super.pretrained()

  override def pretrained(name: String): E5VEmbeddings =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): E5VEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): E5VEmbeddings =
    super.pretrained(name, lang, remoteLoc)

}

trait ReadE5VEmbeddingsDLModel extends ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[E5VEmbeddings] =>
  val suffix: String = "_e5v"
  override val openvinoFile: String = "e5v_openvino"
  def readModel(instance: E5VEmbeddings, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case Openvino.name =>
        val languageModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_language_model-int4.xml"), suffix)

        val visionEmbeddingsModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_vision_embeddings_model.xml"), suffix)

        val textEmbeddingsModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_text_embeddings_model.xml"), suffix)

        val imagePackerModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_image_packer.xml"), suffix)

        val mergeModelWrappers =
          readOpenvinoModels(path, spark, Seq("openvino_multimodal_merger.xml"), suffix)

        val ovWrapper = E5VWrappers(
          languageModel = languageModelWrappers("openvino_language_model-int4.xml"),
          visionEmbeddingsModel =
            visionEmbeddingsModelWrappers("openvino_vision_embeddings_model.xml"),
          textEmbeddingsModel = textEmbeddingsModelWrappers("openvino_text_embeddings_model.xml"),
          mergeModel = mergeModelWrappers("openvino_multimodal_merger.xml"),
          imagePackerModel = imagePackerModelWrappers("openvino_image_packer.xml"))
        val preprocessor = Preprocessor(
          do_normalize = true,
          do_resize = true,
          "E5VFeatureExtractor",
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
      useOpenvino: Boolean = false): E5VEmbeddings = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        isDecoder = false,
        custom = Some(
          List(
            "openvino_language_model-int4",
            "openvino_vision_embeddings_model",
            "openvino_text_embeddings_model",
            "openvino_image_packer",
            "openvino_multimodal_merger")))
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

    val bosTokenId = (modelConfig \ "text_config" \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "text_config" \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "text_config" \ "eos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "text_config" \ "vocab_size").extract[Int]

    val imageToken = (modelConfig \ "image_token_index").extract[Int]
    val imageGridPinpoints: Array[Array[Int]] =
      (modelConfig \ "image_grid_pinpoints").extract[Array[Array[Int]]]
    val imageGridPinpointsMap: Map[Int, Array[Int]] =
      imageGridPinpoints.zipWithIndex.map { case (pinpoints, index) =>
        (index, pinpoints)
      }.toMap
    // Check if tokenizer.json exists
    val tokenizerPath = s"$localModelPath/assets/tokenizer.json"
    val tokenizerExists = new java.io.File(tokenizerPath).exists()
    val (vocabs, addedTokens, bytePairs) = if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      // extract vocab from tokenizer.json ( model -> vocab)
      var vocabs: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]

      // extract merges from tokenizer.json ( model -> merges)
//      val bytePairs = (tokenizerConfig \ "model" \ "merges")
//        .extract[List[Array[String]]]
//        .filter(w => w.length == 2)
//        .map { case Array(c1, c2) => (c1, c2) }
//        .zipWithIndex
//        .toMap
      val bytePairs = (tokenizerConfig \ "model" \ "merges")
        .extract[List[String]]
        .map(_.split(" "))
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

    val annotatorModel = new E5VEmbeddings()
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
      .setImageGridPinpoints(imageGridPinpointsMap)

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
            modelName = "openvino_vision_embeddings_model")
        val textWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_text_embeddings_model")

        val imagePackerModelWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_image_packer")

        val mergeWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_multimodal_merger")
        val languageModelWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine,
            modelName = "openvino_language_model-int4")

        val openvinoWrapper = E5VWrappers(
          languageModel = languageModelWrapper,
          visionEmbeddingsModel = visionWrapper,
          textEmbeddingsModel = textWrapper,
          imagePackerModel = imagePackerModelWrapper,
          mergeModel = mergeWrapper)
        annotatorModel.setModelIfNotSet(spark, preprocessorConfig, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object E5VEmbeddings extends ReadablePretrainedE5VEmbeddings with ReadE5VEmbeddingsDLModel
