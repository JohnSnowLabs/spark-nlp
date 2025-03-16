/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.ml.ai.CLIP
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
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
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForQuestionAnswering
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, CLIPTokenizer}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Zero Shot Image Classifier based on CLIP.
  *
  * CLIP (Contrastive Language-Image Pre-Training) is a neural network that was trained on image
  * and text pairs. It has the ability to predict images without training on any hard-coded
  * labels. This makes it very flexible, as labels can be provided during inference. This is
  * similar to the zero-shot capabilities of the GPT-2 and 3 models.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  *
  * {{{
  * val imageClassifier = CLIPForZeroShotClassification.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("label")
  * }}}
  * The default model is `"zero_shot_classifier_clip_vit_base_patch32"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Zero-Shot+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/CLIPForZeroShotClassificationTestSpec.scala CLIPForZeroShotClassificationTestSpec]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.ImageAssembler
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val imageDF = ResourceHelper.spark.read
  *   .format("image")
  *   .option("dropInvalid", value = true)
  *   .load("src/test/resources/image/")
  *
  * val imageAssembler: ImageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val candidateLabels = Array(
  *   "a photo of a bird",
  *   "a photo of a cat",
  *   "a photo of a dog",
  *   "a photo of a hen",
  *   "a photo of a hippo",
  *   "a photo of a room",
  *   "a photo of a tractor",
  *   "a photo of an ostrich",
  *   "a photo of an ox")
  *
  * val imageClassifier = CLIPForZeroShotClassification
  *   .pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("label")
  *   .setCandidateLabels(candidateLabels)
  *
  * val pipeline =
  *   new Pipeline().setStages(Array(imageAssembler, imageClassifier)).fit(imageDF).transform(imageDF)
  *
  * pipeline
  *   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "label.result")
  *   .show(truncate = false)
  * +-----------------+-----------------------+
  * |image_name       |result                 |
  * +-----------------+-----------------------+
  * |palace.JPEG      |[a photo of a room]    |
  * |egyptian_cat.jpeg|[a photo of a cat]     |
  * |hippopotamus.JPEG|[a photo of a hippo]   |
  * |hen.JPEG         |[a photo of a hen]     |
  * |ostrich.JPEG     |[a photo of an ostrich]|
  * |junco.JPEG       |[a photo of a bird]    |
  * |bluetick.jpg     |[a photo of a dog]     |
  * |chihuahua.jpg    |[a photo of a dog]     |
  * |tractor.JPEG     |[a photo of a tractor] |
  * |ox.JPEG          |[a photo of an ox]     |
  * +-----------------+-----------------------+
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
class CLIPForZeroShotClassification(override val uid: String)
    extends AnnotatorModel[CLIPForZeroShotClassification]
    with HasBatchedAnnotateImage[CLIPForZeroShotClassification]
    with HasImageFeatureProperties
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasEngine
    with HasRescaleFactor {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("CLIPForZeroShotClassification"))

  /** Output annotator type : CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CATEGORY

  /** Input annotator type : IMAGE
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)

  /** Candidate labels for classification, you can set candidateLabels dynamically during the
    * runtime
    *
    * @group param
    */
  val candidateLabels: StringArrayParam = new StringArrayParam(
    this,
    "candidateLabels",
    "Candidate labels for classification, you can set candidateLabels dynamically during the runtime")

  /** @group getParam */
  def getCandidateLabels: Array[String] = $(candidateLabels)

  /** @group setParam */
  def setCandidateLabels(value: Array[String]): this.type = set(candidateLabels, value)

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

  private var _model: Option[Broadcast[CLIP]] = None

  /** @group getParam */
  def getModelIfNotSet: CLIP = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflow: Option[TensorflowWrapper],
      onnx: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper],
      preprocessor: Preprocessor): this.type = {
    if (_model.isEmpty) {

      val tokenizer = BpeTokenizer
        .forModel("clip", merges = getMerges, vocab = getVocabulary)
        .asInstanceOf[CLIPTokenizer]

      _model = Some(
        spark.sparkContext.broadcast(
          new CLIP(
            tensorflow,
            onnx,
            openvinoWrapper,
            configProtoBytes = None,
            tokenizer = tokenizer,
            preprocessor = preprocessor)))
    }
    this
  }

  params
  setDefault(
    batchSize -> 2,
    doNormalize -> true,
    doRescale -> true,
    doResize -> true,
    imageMean -> Array(0.48145466, 0.4578275, 0.40821073),
    imageStd -> Array(0.26862954, 0.26130258, 0.27577711),
    resample -> 2,
    rescaleFactor -> 1 / 255.0d,
    size -> 224)

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

    val nonEmptyImages = imagesWithRow.map(_._1).filter(_.result.nonEmpty).toArray

    val candidateLabels: Array[String] = getCandidateLabels

    val allAnnotations =
      if (nonEmptyImages.nonEmpty) {
        getModelIfNotSet.predict(
          images = nonEmptyImages,
          labels = candidateLabels,
          batchSize = $(batchSize))
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
    super.onWrite(path, spark)
    getEngine match {
      case TensorFlow.name =>
        throw new Exception("Tensorflow is currently not supported by this annotator.")
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrapper.get
        writeOnnxModel(
          path,
          spark,
          wrappers,
          CLIPForZeroShotClassification.suffix,
          CLIPForZeroShotClassification.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          CLIPForZeroShotClassification.openvinoFile)
    }
  }
}

trait ReadablePretrainedCLIPForZeroShotClassificationModel
    extends ParamsAndFeaturesReadable[CLIPForZeroShotClassification]
    with HasPretrained[CLIPForZeroShotClassification] {
  override val defaultModelName: Some[String] = Some("zero_shot_classifier_clip_vit_base_patch32")

  /** Java compliant-overrides */
  override def pretrained(): CLIPForZeroShotClassification = super.pretrained()

  override def pretrained(name: String): CLIPForZeroShotClassification =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): CLIPForZeroShotClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): CLIPForZeroShotClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCLIPForZeroShotClassificationModel
    extends ReadTensorflowModel
    with ReadOnnxModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[CLIPForZeroShotClassification] =>

  override val tfFile: String = "clip_classification_tensorflow"
  override val onnxFile: String = "clip_classification_onnx"
  override val openvinoFile: String = "clip_classification_openvino"
  val suffix: String = "_clip_classification"

  def readModel(
      instance: CLIPForZeroShotClassification,
      path: String,
      spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = instance.getDoNormalize,
      do_resize = instance.getDoRescale,
      feature_extractor_type = "CLIPFeatureExtractor",
      image_mean = instance.getImageMean,
      image_std = instance.getImageStd,
      resample = instance.getResample,
      do_rescale = instance.getDoRescale,
      rescale_factor = instance.getRescaleFactor,
      size = instance.getSize)

    instance.getEngine match {
      case TensorFlow.name =>
        throw new Exception("Tensorflow is currently not supported by this annotator.")
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, CLIPForZeroShotClassification.suffix)

        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None, preprocessor)
      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, CLIPForZeroShotClassification.suffix)
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper), preprocessor)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  /** Loads a local SavedModel file of the model. For CLIP, requires also image preprocessor
    * config and vocab file.
    *
    * @param modelPath
    *   Path of the Model
    * @param spark
    *   Spark Instance
    * @return
    */
  def loadSavedModel(modelPath: String, spark: SparkSession): CLIPForZeroShotClassification = {
    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

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

    val annotatorModel = new CLIPForZeroShotClassification()
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

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        throw new Exception("Tensorflow is currently not supported by this annotator.")
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None, preprocessorConfig)
      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper), preprocessorConfig)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

/** This is the companion object of [[CLIPForZeroShotClassification]]. Please refer to that class
  * for the documentation.
  */
object CLIPForZeroShotClassification
    extends ReadablePretrainedCLIPForZeroShotClassificationModel
    with ReadCLIPForZeroShotClassificationModel
