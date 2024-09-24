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

package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.ml.ai.VisionEncoderDecoder
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, Gpt2Tokenizer}
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.parse
import org.json4s.{DefaultFormats, JValue}

/** VisionEncoderDecoder model that converts images into text captions. It allows for the use of
  * pretrained vision auto-encoding models, such as ViT, BEiT, or DeiT as the encoder, in
  * combination with pretrained language models, like RoBERTa, GPT2, or BERT as the decoder.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  *
  * {{{
  * val imageClassifier = VisionEncoderDecoderForImageCaptioning.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("caption")
  * }}}
  * The default model is `"image_captioning_vit_gpt2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Image+Captioning Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/VisionEncoderDecoderForImageCaptioningTestSpec.scala VisionEncoderDecoderTestSpec]].
  *
  * '''Note:'''
  *
  * This is a very computationally expensive module especially on larger batch sizes. The use of
  * an accelerator such as GPU is recommended.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.ImageAssembler
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF: DataFrame = spark.read
  *   .format("image")
  *   .option("dropInvalid", value = true)
  *   .load("src/test/resources/image/")
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val imageCaptioning = VisionEncoderDecoderForImageCaptioning
  *   .pretrained()
  *   .setBeamSize(2)
  *   .setDoSample(false)
  *   .setInputCols("image_assembler")
  *   .setOutputCol("caption")
  *
  * val pipeline = new Pipeline().setStages(Array(imageAssembler, imageCaptioning))
  * val pipelineDF = pipeline.fit(imageDF).transform(imageDF)
  *
  * pipelineDF
  *   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "caption.result")
  *   .show(truncate = false)
  *
  * +-----------------+---------------------------------------------------------+
  * |image_name       |result                                                   |
  * +-----------------+---------------------------------------------------------+
  * |palace.JPEG      |[a large room filled with furniture and a large window]  |
  * |egyptian_cat.jpeg|[a cat laying on a couch next to another cat]            |
  * |hippopotamus.JPEG|[a brown bear in a body of water]                        |
  * |hen.JPEG         |[a flock of chickens standing next to each other]        |
  * |ostrich.JPEG     |[a large bird standing on top of a lush green field]     |
  * |junco.JPEG       |[a small bird standing on a wet ground]                  |
  * |bluetick.jpg     |[a small dog standing on a wooden floor]                 |
  * |chihuahua.jpg    |[a small brown dog wearing a blue sweater]               |
  * |tractor.JPEG     |[a man is standing in a field with a tractor]            |
  * |ox.JPEG          |[a large brown cow standing on top of a lush green field]|
  * +-----------------+---------------------------------------------------------+
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
class VisionEncoderDecoderForImageCaptioning(override val uid: String)
    extends AnnotatorModel[VisionEncoderDecoderForImageCaptioning]
    with HasBatchedAnnotateImage[VisionEncoderDecoderForImageCaptioning]
    with HasImageFeatureProperties
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine
    with HasRescaleFactor
    with HasGeneratorProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("VisionEncoderDecoderForImageCaptioning"))

  /** Output annotator type : CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** Input annotator type : IMAGE
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): this.type =
    set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
    *
    * @group param
    */
  protected[nlp] val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  protected[nlp] def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** @group getParam */
  protected[nlp] def getVocabulary: Map[String, Int] = $$(vocabulary)

  /** Holding merges.txt for BPE Tokenization
    *
    * @group param
    */
  protected[nlp] val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges")

  /** @group setParam */
  protected[nlp] def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** @group getParam */
  protected[nlp] def getMerges: Map[(String, String), Int] = $$(merges)

  protected[nlp] val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  protected[nlp] def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  protected[nlp] def getGenerationConfig: GenerationConfig = $$(generationConfig)

  private var _model: Option[Broadcast[VisionEncoderDecoder]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[EncoderDecoderWithoutPastWrappers],
      preprocessor: Preprocessor): this.type = {
    if (_model.isEmpty) {

      val tokenizer = BpeTokenizer
        .forModel("gpt2", merges = getMerges, vocab = getVocabulary)
        .asInstanceOf[Gpt2Tokenizer]

      _model = Some(
        spark.sparkContext.broadcast(
          new VisionEncoderDecoder(
            tensorflowWrapper,
            onnxWrapper,
            configProtoBytes = getConfigProtoBytes,
            tokenizer = tokenizer,
            preprocessor = preprocessor,
            signatures = getSignatures,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: VisionEncoderDecoder = _model.get.value

  setDefault(
    batchSize -> 2,
    beamSize -> 1,
    doNormalize -> true,
    doRescale -> true,
    doResize -> true,
    doSample -> true,
    imageMean -> Array(0.5d, 0.5d, 0.5d),
    imageStd -> Array(0.5d, 0.5d, 0.5d),
    maxOutputLength -> 50,
    minOutputLength -> 0,
    nReturnSequences -> 1,
    noRepeatNgramSize -> 0,
    repetitionPenalty -> 1.0,
    resample -> 2,
    rescaleFactor -> 1 / 255.0d,
    size -> 224,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0)

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
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

    // Zip annotations to the row it belongs to
    val imagesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) => annotations.map(x => (x, i)) }

    val noneEmptyImages = imagesWithRow.map(_._1).filter(_.result.nonEmpty).toArray

    val allAnnotations =
      if (noneEmptyImages.nonEmpty) {
        getModelIfNotSet.generateFromImage(
          images = noneEmptyImages,
          batchSize = $(batchSize),
          maxOutputLength = getMaxOutputLength,
          minOutputLength = getMinOutputLength,
          doSample = getDoSample,
          beamSize = getBeamSize,
          numReturnSequences = getNReturnSequences,
          temperature = getTemperature,
          topK = getTopK,
          topP = getTopP,
          repetitionPenalty = getRepetitionPenalty,
          noRepeatNgramSize = getNoRepeatNgramSize,
          randomSeed = getRandomSeed)
      } else {
        Seq.empty[Annotation]
      }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = allAnnotations
        // zip each annotation with its corresponding row index
        .zip(imagesWithRow)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowAnnotations.nonEmpty)
        rowAnnotations
      else
        Seq.empty[Annotation]
    })

  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          VisionEncoderDecoderForImageCaptioning.suffix,
          VisionEncoderDecoderForImageCaptioning.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers.get
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.encoder, "encoder_model.onnx")),
          VisionEncoderDecoderForImageCaptioning.suffix)
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.decoder, "decoder_model.onnx")),
          VisionEncoderDecoderForImageCaptioning.suffix)

    }
  }
}

trait ReadablePretrainedVisionEncoderDecoderModel
    extends ParamsAndFeaturesReadable[VisionEncoderDecoderForImageCaptioning]
    with HasPretrained[VisionEncoderDecoderForImageCaptioning] {
  override val defaultModelName: Some[String] = Some("image_captioning_vit_gpt2")

  /** Java compliant-overrides */
  override def pretrained(): VisionEncoderDecoderForImageCaptioning = super.pretrained()

  override def pretrained(name: String): VisionEncoderDecoderForImageCaptioning =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): VisionEncoderDecoderForImageCaptioning =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): VisionEncoderDecoderForImageCaptioning =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadVisionEncoderDecoderDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[VisionEncoderDecoderForImageCaptioning] =>

  override val tfFile: String = "vision_encoder_decoder_tensorflow"
  override val onnxFile: String = "vision_encoder_decoder_onnx"
  val suffix = "_image_classification"
  def readModel(
      instance: VisionEncoderDecoderForImageCaptioning,
      path: String,
      spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = instance.getDoNormalize,
      do_resize = instance.getDoRescale,
      feature_extractor_type = "ViTFeatureExtractor", // Default Extractor
      image_mean = instance.getImageMean,
      image_std = instance.getImageStd,
      resample = instance.getResample,
      do_rescale = instance.getDoRescale,
      rescale_factor = instance.getRescaleFactor,
      size = instance.getSize)

    instance.getEngine match {
      case TensorFlow.name =>
        val tf = readTensorflowModel(path, spark, "_vision_encoder_decoder_tf")
        instance.setModelIfNotSet(spark, Some(tf), None, preprocessor)

      case ONNX.name =>
        val wrappers =
          readOnnxModels(
            path,
            spark,
            Seq("encoder_model.onnx", "decoder_model.onnx"),
            VisionEncoderDecoderForImageCaptioning.suffix,
            dataFilePostfix = ".onnx_data")

        val onnxWrappers = EncoderDecoderWithoutPastWrappers(
          wrappers("encoder_model.onnx"),
          decoder = wrappers("decoder_model.onnx"))

        instance.setModelIfNotSet(spark, None, Some(onnxWrappers), preprocessor)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  /** Loads a local SavedModel file of the model. For VisionEncoderDecoder, requires also image
    * preprocessor config and vocab file.
    *
    * @param modelPath
    *   Path of the Model
    * @param spark
    *   Spark Instance
    * @return
    */
  def loadSavedModel(
      modelPath: String,
      spark: SparkSession): VisionEncoderDecoderForImageCaptioning = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath, isEncoderDecoder = true)

    val vocab = {
      val json = loadJsonStringAsset(localModelPath, "vocab.json")
      JsonParser.parseObject[Map[String, Int]](json)
    }

    val merges = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2 && !w.startsWith("#"))
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val generationConfig = {
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

      val bosTokenId = (modelConfig \ "decoder_start_token_id").extract[Int]
      val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
      val padTokenId = (modelConfig \ "pad_token_id").extract[Int]
      val vocabSize = (modelConfig \ "decoder" \ "vocab_size").extract[Int]

      def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
        if (array.nonEmpty) Some(array) else None

      GenerationConfig(
        bosTokenId,
        padTokenId,
        eosTokenId,
        vocabSize,
        arrayOrNone(beginSuppressTokens),
        arrayOrNone(suppressTokenIds),
        arrayOrNone(forcedDecoderIds))
    }

    /*Universal parameters for all engines*/
    val annotatorModel = new VisionEncoderDecoderForImageCaptioning()
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setDoResize(preprocessorConfig.do_resize)
      .setFeatureExtractorType(preprocessorConfig.feature_extractor_type)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setSize(preprocessorConfig.size)
      .setDoRescale(preprocessorConfig.do_rescale)
      .setRescaleFactor(preprocessorConfig.rescale_factor)
      .setVocabulary(vocab)
      .setMerges(merges)
      .setGenerationConfig(generationConfig)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (tfWrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(tfWrapper), None, preprocessorConfig)

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
          EncoderDecoderWithoutPastWrappers(onnxWrapperEncoder, onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrappers), preprocessorConfig)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[VisionEncoderDecoderForImageCaptioning]]. Please refer to
  * that class for the documentation.
  */
object VisionEncoderDecoderForImageCaptioning
    extends ReadablePretrainedVisionEncoderDecoderModel
    with ReadVisionEncoderDecoderDLModel
