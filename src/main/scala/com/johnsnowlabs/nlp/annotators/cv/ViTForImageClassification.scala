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

import com.johnsnowlabs.ml.ai.ViTClassifier
import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntArrayParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** Vision Transformer (ViT) for image classification.
  *
  * ViT is a transformer based alternative to the convolutional neural networks usually used for
  * image recognition tasks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val imageClassifier = ViTForImageClassification.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("class")
  * }}}
  * The default model is `"image_classifier_vit_base_patch16_224"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Image+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/ViTImageClassificationTestSpec.scala ViTImageClassificationTestSpec]].
  *
  * '''References:'''
  *
  * [[https://arxiv.org/abs/2010.11929 An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]]
  *
  * '''Paper Abstract:'''
  *
  * ''While the Transformer architecture has become the de-facto standard for natural language
  * processing tasks, its applications to computer vision remain limited. In vision, attention is
  * either applied in conjunction with convolutional networks, or used to replace certain
  * components of convolutional networks while keeping their overall structure in place. We show
  * that this reliance on CNNs is not necessary and a pure transformer applied directly to
  * sequences of image patches can perform very well on image classification tasks. When
  * pre-trained on large amounts of data and transferred to multiple mid-sized or small image
  * recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains
  * excellent results compared to state-of-the-art convolutional networks while requiring
  * substantially fewer computational resources to train.''
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
  * val imageClassifier = ViTForImageClassification
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
  * |egyptian_cat.jpeg|[Egyptian cat]                                            |
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
class ViTForImageClassification(override val uid: String)
    extends AnnotatorModel[ViTForImageClassification]
    with HasBatchedAnnotateImage[ViTForImageClassification]
    with HasImageFeatureProperties
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("ViTForImageClassification"))

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
  def setConfigProtoBytes(bytes: Array[Int]): ViTForImageClassification.this.type =
    set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Labels used to decode predicted IDs back to string tags
    *
    * @group param
    */
  val labels: MapFeature[String, BigInt] = new MapFeature(this, "labels").setProtected()

  /** @group setParam */
  def setLabels(value: Map[String, BigInt]): this.type = set(labels, value)

  /** Returns labels used to train this model */
  def getClasses: Array[String] = {
    $$(labels).keys.toArray
  }

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

  private var _model: Option[Broadcast[ViTClassifier]] = None

  /** @group getParam */
  def getModelIfNotSet: ViTClassifier = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      preprocessor: Preprocessor): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new ViTClassifier(
            tensorflowWrapper,
            onnxWrapper,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            preprocessor = preprocessor,
            signatures = getSignatures)))
    }
    this
  }

  setDefault(batchSize -> 2)

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
            size = getSize))
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
    val suffix = "_image_classification"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          ViTForImageClassification.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          ViTForImageClassification.onnxFile)
    }
  }

}

trait ReadablePretrainedViTForImageModel
    extends ParamsAndFeaturesReadable[ViTForImageClassification]
    with HasPretrained[ViTForImageClassification] {
  override val defaultModelName: Some[String] = Some("image_classifier_vit_base_patch16_224")

  /** Java compliant-overrides */
  override def pretrained(): ViTForImageClassification = super.pretrained()

  override def pretrained(name: String): ViTForImageClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): ViTForImageClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): ViTForImageClassification = super.pretrained(name, lang, remoteLoc)
}

trait ReadViTForImageDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[ViTForImageClassification] =>

  override val tfFile: String = "image_classification_tensorflow"
  override val onnxFile: String = "image_classification_onnx"

  def readModel(instance: ViTForImageClassification, path: String, spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = true,
      do_resize = true,
      "ViTFeatureExtractor",
      instance.getImageMean,
      instance.getImageStd,
      instance.getResample,
      instance.getSize)
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

  def loadSavedModel(modelPath: String, spark: SparkSession): ViTForImageClassification = {

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

    /*Universal parameters for all engines*/
    val annotatorModel = new ViTForImageClassification()
      .setLabels(labelJsonMap)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setDoResize(preprocessorConfig.do_resize)
      .setFeatureExtractorType(preprocessorConfig.feature_extractor_type)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setSize(preprocessorConfig.size)

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

/** This is the companion object of [[ViTForImageClassification]]. Please refer to that class for
  * the documentation.
  */
object ViTForImageClassification
    extends ReadablePretrainedViTForImageModel
    with ReadViTForImageDLModel
