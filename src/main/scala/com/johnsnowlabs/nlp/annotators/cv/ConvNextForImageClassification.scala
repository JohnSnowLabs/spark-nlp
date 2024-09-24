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

import com.johnsnowlabs.ml.ai.ConvNextClassifier
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowWrapper}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.DoubleParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** ConvNextForImageClassification is an image classifier based on ConvNet models.
  *
  * The ConvNeXT model was proposed in A ConvNet for the 2020s by Zhuang Liu, Hanzi Mao, Chao-Yuan
  * Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. ConvNeXT is a pure convolutional
  * model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform
  * them.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val imageClassifier = ConvNextForImageClassification.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("class")
  * }}}
  * The default model is `"image_classifier_convnext_tiny_224_local"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Image+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ConvNextForImageClassificationTestSpec.scala ConvNextForImageClassificationTestSpec]].
  *
  * '''References:'''
  *
  * [[https://arxiv.org/abs/2201.03545 A ConvNet for the 2020s]]
  *
  * '''Paper Abstract:'''
  *
  * ''The "Roaring 20s" of visual recognition began with the introduction of Vision Transformers
  * (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model.
  * A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision
  * tasks such as object detection and semantic segmentation. It is the hierarchical Transformers
  * (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers
  * practically viable as a generic vision backbone and demonstrating remarkable performance on a
  * wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still
  * largely credited to the intrinsic superiority of Transformers, rather than the inherent
  * inductive biases of convolutions. In this work, we reexamine the design spaces and test the
  * limits of what a pure ConvNet can achieve. We gradually "modernize" a standard ResNet toward
  * the design of a vision Transformer, and discover several key components that contribute to the
  * performance difference along the way. The outcome of this exploration is a family of pure
  * ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts
  * compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8%
  * ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K
  * segmentation, while maintaining the simplicity and efficiency of standard ConvNets. ''
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
  * val imageClassifier = ConvNextForImageClassification
  *   .pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("class")
  *
  * val pipeline = new Pipeline().setStages(Array(imageAssembler, imageClassifier))
  * val pipelineDF = pipeline.fit(imageDF).transform(imageDF)
  *
  * pipelineDF
  *   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "class.result")
  *   .show(truncate = false)
  * +-----------------+----------------------------------------------------------+
  * |image_name       |result                                                    |
  * +-----------------+----------------------------------------------------------+
  * |palace.JPEG      |[palace]                                                  |
  * |egyptian_cat.jpeg|[tabby, tabby cat]                                        |
  * |hippopotamus.JPEG|[hippopotamus, hippo, river horse, Hippopotamus amphibius]|
  * |hen.JPEG         |[hen]                                                     |
  * |ostrich.JPEG     |[ostrich, Struthio camelus]                               |
  * |junco.JPEG       |[junco, snowbird]                                         |
  * |bluetick.jpg     |[bluetick]                                                |
  * |chihuahua.jpg    |[Chihuahua]                                               |
  * |tractor.JPEG     |[tractor]                                                 |
  * |ox.JPEG          |[ox]                                                      |
  * +-----------------+----------------------------------------------------------+
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
class ConvNextForImageClassification(override val uid: String)
    extends SwinForImageClassification(uid) {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("ConvNextForImageClassification"))

  /** Determines rescale and crop percentage for images smaller than the configured size (Default:
    * `224 / 256d`).
    *
    * If the image size is smaller than the specified size, the smaller edge of the image will be
    * matched to `int(size / cropPct)`. Afterwards the image is cropped to `(size, size)`.
    *
    * @group param
    */
  val cropPct =
    new DoubleParam(this, "cropPct", "Percentage of the resized image to crop")

  /** @group setParam */
  def setCropPct(value: Double): this.type = set(this.cropPct, value)

  /** @group getParam */
  def getCropPct: Double = $(cropPct)

  setDefault(
    batchSize -> 2,
    doNormalize -> true,
    doRescale -> true,
    doResize -> true,
    imageMean -> Array(0.485d, 0.456d, 0.406d),
    imageStd -> Array(0.229d, 0.224d, 0.225d),
    resample -> 3,
    size -> 224,
    rescaleFactor -> 1 / 255d,
    cropPct -> 224 / 256d)

  private var _model: Option[Broadcast[ConvNextClassifier]] = None

  /** @group getParam */
  override def getModelIfNotSet: ConvNextClassifier = _model.get.value

  override def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      preprocessor: Preprocessor): ConvNextForImageClassification.this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new ConvNextClassifier(
            tensorflowWrapper,
            onnxWrapper,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            preprocessor = preprocessor,
            signatures = getSignatures)))
    }
    this
  }

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
        getModelIfNotSet.predict(
          images = noneEmptyImages,
          batchSize = $(batchSize),
          preprocessor = Preprocessor(
            do_normalize = getDoNormalize,
            do_resize = getDoResize,
            feature_extractor_type = getFeatureExtractorType,
            image_mean = getImageMean,
            image_std = getImageStd,
            resample = getResample,
            size = getSize,
            do_rescale = getDoRescale,
            rescale_factor = getRescaleFactor,
            crop_pct = Option(getCropPct)))
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
    val suffix = "_image_classification"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          ConvNextForImageClassification.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          ConvNextForImageClassification.onnxFile)
    }
  }

}

trait ReadablePretrainedConvNextForImageModel
    extends ParamsAndFeaturesReadable[ConvNextForImageClassification]
    with HasPretrained[ConvNextForImageClassification] {
  override val defaultModelName: Some[String] = Some("image_classifier_convnext_tiny_224_local")

  /** Java compliant-overrides */
  override def pretrained(): ConvNextForImageClassification = super.pretrained()

  override def pretrained(name: String): ConvNextForImageClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): ConvNextForImageClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): ConvNextForImageClassification = super.pretrained(name, lang, remoteLoc)
}

trait ReadConvNextForImageDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[ConvNextForImageClassification] =>

  override val tfFile: String = "image_classification_convnext_tensorflow"
  override val onnxFile: String = "image_classification_convnext_onnx"

  def readModel(
      instance: ConvNextForImageClassification,
      path: String,
      spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = instance.getDoNormalize,
      do_resize = instance.getDoRescale,
      feature_extractor_type = "ConvNextFeatureExtractor",
      image_mean = instance.getImageMean,
      image_std = instance.getImageStd,
      resample = instance.getResample,
      do_rescale = instance.getDoRescale,
      rescale_factor = instance.getRescaleFactor,
      size = instance.getSize,
      crop_pct = Option(instance.getCropPct))
    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper =
          readTensorflowModel(path, spark, tfFile, initAllTables = false)

        instance.setModelIfNotSet(spark, Some(tfWrapper), None, preprocessor)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, onnxFile, zipped = true, useBundle = false, None)

        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), preprocessor)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)
  def loadSavedModel(modelPath: String, spark: SparkSession): ConvNextForImageClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    // TODO: sometimes results in [String, BigInt] where BigInt is actually a string
    val labelJsonContent = loadJsonStringAsset(localModelPath, "labels.json")
    val labelJsonMap =
      parse(labelJsonContent, useBigIntForLong = true).values
        .asInstanceOf[Map[String, BigInt]]

    val preprocessorConfigJsonContent =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonContent)

    require(
      preprocessorConfig.size >= 384 || preprocessorConfig.crop_pct.nonEmpty,
      "Property \'crop_pct\' should be defined, if size < 384.")
    val cropPct = preprocessorConfig.crop_pct.get

    val annotatorModel = new ConvNextForImageClassification()
      .setLabels(labelJsonMap)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setDoResize(preprocessorConfig.do_resize)
      .setFeatureExtractorType(preprocessorConfig.feature_extractor_type)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setSize(preprocessorConfig.size)
      .setDoRescale(preprocessorConfig.do_rescale)
      .setRescaleFactor(preprocessorConfig.rescale_factor)
      .setCropPct(cropPct)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (tfwrapper, signatures) =
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
          .setModelIfNotSet(spark, Some(tfwrapper), None, preprocessorConfig)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)

        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), preprocessorConfig)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[ConvNextForImageClassification]]. Please refer to that class
  * for the documentation.
  */
object ConvNextForImageClassification
    extends ReadablePretrainedConvNextForImageModel
    with ReadConvNextForImageDLModel
