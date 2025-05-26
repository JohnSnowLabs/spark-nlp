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

/** Florence2: Advancing a Unified Representation for a Variety of Vision Tasks
  *
  * Florence-2 is an advanced vision foundation model from Microsoft that uses a prompt-based
  * approach to handle a wide range of vision and vision-language tasks. It can interpret simple
  * text prompts to perform tasks like captioning, object detection, segmentation, OCR, and more.
  * The model leverages the FLD-5B dataset, containing 5.4 billion annotations across 126 million
  * images, to master multi-task learning. Its sequence-to-sequence architecture enables it to
  * excel in both zero-shot and fine-tuned settings.
  *
  * Pretrained and finetuned models can be loaded with `pretrained` of the companion object: {{ {
  * val florence2 = Florence2Transformer.pretrained() .setInputCols("image")
  * .setOutputCol("generation") }} } The default model is `"florence2_base_ft_int4"`, if no name
  * is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Vision+Tasks Models Hub]].
  *
  * ==Supported Tasks==
  *
  * Florence-2 supports a variety of tasks through prompt engineering. The following prompt tokens
  * can be used:
  *
  *   - <CAPTION>: Image captioning
  *   - <DETAILED_CAPTION>: Detailed image captioning
  *   - <MORE_DETAILED_CAPTION>: Paragraph-level captioning
  *   - <CAPTION_TO_PHRASE_GROUNDING>: Phrase grounding from caption (requires additional text
  *     input)
  *   - <OD>: Object detection
  *   - <DENSE_REGION_CAPTION>: Dense region captioning
  *   - <REGION_PROPOSAL>: Region proposal
  *   - <OCR>: Optical Character Recognition (plain text extraction)
  *   - <OCR_WITH_REGION>: OCR with region information
  *   - <REFERRING_EXPRESSION_SEGMENTATION>: Segmentation for a referred phrase (requires
  *     additional text input)
  *   - <REGION_TO_SEGMENTATION>: Polygon mask for a region (requires additional text input)
  *   - <OPEN_VOCABULARY_DETECTION>: Open vocabulary detection for a phrase (requires additional
  *     text input)
  *   - <REGION_TO_CATEGORY>: Category of a region (requires additional text input)
  *   - <REGION_TO_DESCRIPTION>: Description of a region (requires additional text input)
  *   - <REGION_TO_OCR>: OCR for a region (requires additional text input)
  *
  * ==Example Usage==
  *
  * {{ { import com.johnsnowlabs.nlp.base.ImageAssembler import
  * com.johnsnowlabs.nlp.annotators.cv.Florence2Transformer import org.apache.spark.ml.Pipeline
  *
  * val imageAssembler = new ImageAssembler() .setInputCol("image")
  * .setOutputCol("image_assembler")
  *
  * val florence2 = Florence2Transformer.pretrained("florence2_base_ft_int4")
  * .setInputCols("image_assembler") .setOutputCol("answer") .setMaxOutputLength(50)
  *
  * val pipeline = new Pipeline().setStages(Array(imageAssembler, florence2))
  *
  * val data = Seq("/path/to/image.jpg").toDF("image") val result =
  * pipeline.fit(data).transform(data) result.select("answer.result").show(truncate = false) }} }
  *
  * ==References==
  *
  *   - Florence-2 technical report: https://arxiv.org/abs/2311.06242
  *   - Hugging Face model card: https://huggingface.co/microsoft/Florence-2-base-ft
  *   - Official sample notebook:
  *     https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
  *
  * For more details and advanced usage, see the official documentation and sample notebooks.
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
          Seq((wrappers.get.modelMergerModel, "merger_model.xml")),
          PaliGemmaForMultiModal.suffix)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }
}

trait ReadablePretrainedFlorence2TransformerModel
    extends ParamsAndFeaturesReadable[Florence2Transformer]
    with HasPretrained[Florence2Transformer] {
  override val defaultModelName: Some[String] = Some("florence2_base_ft_int4")

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
          readOpenvinoModels(path, spark, Seq("merger_model.xml"), suffix)
        val ovWrapper = {
          Florence2Wrappers(
            encoderModel = encoderWrappers("encoder.xml"),
            decoderModel = decoderWrappers("decoder.xml"),
            textEmbeddingsModel = textEmbeddingsWrappers("text_embedding.xml"),
            imageEmbedModel = imageEmbeddingsWrappers("image_embedding.xml"),
            modelMergerModel = modelMergerWrappers("merger_model.xml"))
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
          Some(List("encoder", "decoder", "text_embedding", "merger_model", "image_embedding")))
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
            modelName = "merger_model")
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
