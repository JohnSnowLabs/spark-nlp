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

import com.johnsnowlabs.ml.ai.BLIPClassifier
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
import com.johnsnowlabs.ml.util.TensorFlow
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BertTokenizer, SpecialTokens}
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** BLIPForQuestionAnswering can load BLIP models for visual question answering. The model
  * consists of a vision encoder, a text encoder as well as a text decoder. The vision encoder
  * will encode the input image, the text encoder will encode the input question together with the
  * encoding of the image, and the text decoder will output the answer to the question.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val visualQAClassifier = BLIPForQuestionAnswering.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("answer")
  * }}}
  * The default model is `"blip_vqa_base"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Question+Answering Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/BLIPForQuestionAnsweringTest.scala]].
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
  * val testDF: DataFrame = imageDF.withColumn("text", lit("What's this picture about?"))
  *
  * val imageAssembler: ImageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val visualQAClassifier = BLIPForQuestionAnswering.pretrained()
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
  * |[file:///content/images/cat_image.jpg]|[cats]|
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

class BLIPForQuestionAnswering(override val uid: String)
    extends AnnotatorModel[BLIPForQuestionAnswering]
    with HasBatchedAnnotateImage[BLIPForQuestionAnswering]
    with HasImageFeatureProperties
    with WriteTensorflowModel
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("BLIPForQuestionAnswering"))

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

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
  def setConfigProtoBytes(bytes: Array[Int]): BLIPForQuestionAnswering.this.type =
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
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** @group getParam */
  protected[nlp] def getVocabulary: Map[String, Int] = $$(vocabulary)

  /** Max sentence length to process (Default: `512`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  private var _model: Option[Broadcast[BLIPClassifier]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      preprocessor: Preprocessor,
      tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      val specialTokens = SpecialTokens.getSpecialTokensForModel("bert", getVocabulary)
      val bertTokenizer = new BertTokenizer(getVocabulary, specialTokens)

      _model = Some(
        spark.sparkContext.broadcast(
          new BLIPClassifier(
            tensorflow,
            configProtoBytes = getConfigProtoBytes,
            tokenizer = bertTokenizer,
            preprocessor = preprocessor,
            signatures = getSignatures,
            vocabulary = $$(vocabulary))))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: BLIPClassifier = _model.get.value

  setDefault(batchSize -> 8, size -> 384, maxSentenceLength -> 50)

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
      .filter { annotationImages =>
        annotationImages.exists(_.text.nonEmpty)
      }
      .map { cleanAnnotationImages =>
        val validImages = cleanAnnotationImages.filter(_.result.nonEmpty)
        val questionAnnotations = extractInputAnnotation(validImages)

        getModelIfNotSet.predict(
          validImages,
          questionAnnotations,
          $(batchSize),
          $(maxSentenceLength))
      }
  }

  private def extractInputAnnotation(
      annotationImages: Array[AnnotationImage]): Seq[Annotation] = {
    val questions = annotationImages.map(annotationImage => Annotation(annotationImage.text))
    val sentenceAnnotations =
      new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val sentencesQuestions = sentenceAnnotations.annotate(questions)

    val tokenizerAnnotation = new RegexTokenizer().setInputCols("sentence").setOutputCol("token")
    val tokenQuestions = tokenizerAnnotation.annotate(sentencesQuestions)

    sentencesQuestions ++ tokenQuestions
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper,
      "_image_qa",
      BLIPForQuestionAnswering.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedBLIPForQuestionAnswering
    extends ParamsAndFeaturesReadable[BLIPForQuestionAnswering]
    with HasPretrained[BLIPForQuestionAnswering] {

  override val defaultModelName: Some[String] = Some("blip_vqa_base")

  /** Java compliant-overrides */
  override def pretrained(): BLIPForQuestionAnswering = super.pretrained()

  override def pretrained(name: String): BLIPForQuestionAnswering =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): BLIPForQuestionAnswering =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): BLIPForQuestionAnswering =
    super.pretrained(name, lang, remoteLoc)

}

trait ReadBLIPForQuestionAnsweringDLModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[BLIPForQuestionAnswering] =>
  override val tfFile: String = "blip_vqa_tensorflow"

  def readModel(instance: BLIPForQuestionAnswering, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_blip_vqa_tf", initAllTables = false)

    val preprocessor = Preprocessor(
      do_normalize = true,
      do_resize = true,
      "BLIPFeatureExtractor",
      instance.getImageMean,
      instance.getImageStd,
      instance.getResample,
      instance.getSize)

    instance.setModelIfNotSet(spark, preprocessor, tf)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BLIPForQuestionAnswering = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig = Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)
    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val annotatorModel = new BLIPForQuestionAnswering()
    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setVocabulary(vocabs)
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, preprocessorConfig, wrapper)
          .setSize(384)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object BLIPForQuestionAnswering
    extends ReadablePretrainedBLIPForQuestionAnswering
    with ReadBLIPForQuestionAnsweringDLModel
