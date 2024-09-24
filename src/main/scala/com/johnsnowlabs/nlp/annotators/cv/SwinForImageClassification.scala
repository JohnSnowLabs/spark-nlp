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
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** SwinImageClassification is an image classifier based on Swin.
  *
  * The Swin Transformer was proposed in Swin Transformer: Hierarchical Vision Transformer using
  * Shifted Windows by Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin,
  * Baining Guo.
  *
  * It is basically a hierarchical Transformer whose representation is computed with shifted
  * windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
  * computation to non-overlapping local windows while also allowing for cross-window connection.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val imageClassifier = SwinForImageClassification.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("class")
  * }}}
  * The default model is `"image_classifier_swin_base_patch4_window7_224"`, if no name is
  * provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Image+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/cv/SwinForImageClassificationTest.scala SwinForImageClassificationTest]].
  *
  * '''References:'''
  *
  * [[https://arxiv.org/pdf/2103.14030.pdf Swin Transformer: Hierarchical Vision Transformer using Shifted Windows]]
  *
  * '''Paper Abstract:'''
  *
  * ''This paper presents a new vision Transformer, called Swin Transformer, that capably serves
  * as a general-purpose backbone for computer vision. Challenges in adapting Transformer from
  * language to vision arise from differences between the two domains, such as large variations in
  * the scale of visual entities and the high resolution of pixels in images compared to words in
  * text. To address these differences, we propose a hierarchical Transformer whose representation
  * is computed with Shifted windows. The shifted windowing scheme brings greater efficiency by
  * limiting self-attention computation to non-overlapping local windows while also allowing for
  * cross-window connection. This hierarchical architecture has the flexibility to model at
  * various scales and has linear computational complexity with respect to image size. These
  * qualities of Swin Transformer make it compatible with a broad range of vision tasks, including
  * image classification (87.3 top-1 accuracy on ImageNet-1K) and dense prediction tasks such as
  * object detection (58.7 box AP and 51.1 mask AP on COCO test- dev) and semantic segmentation
  * (53.5 mIoU on ADE20K val). Its performance surpasses the previous state-of-the- art by a large
  * margin of +2.7 box AP and +2.6 mask AP on COCO, and +3.2 mIoU on ADE20K, demonstrating the
  * potential of Transformer-based models as vision backbones. The hierarchical design and the
  * shifted window approach also prove beneficial for all-MLP architectures.''
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
  * val imageClassifier = SwinForImageClassification
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
class SwinForImageClassification(override val uid: String)
    extends ViTForImageClassification(uid)
    with HasRescaleFactor {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("SwinForImageClassification"))

  setDefault(
    batchSize -> 2,
    doNormalize -> true,
    doRescale -> true,
    doResize -> true,
    imageMean -> Array(0.485d, 0.456d, 0.406d),
    imageStd -> Array(0.229d, 0.224d, 0.225d),
    resample -> 3,
    size -> 224,
    rescaleFactor -> 1 / 255.0d)

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
            rescale_factor = getRescaleFactor))
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
          SwinForImageClassification.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          SwinForImageClassification.onnxFile)
    }
  }

}

trait ReadablePretrainedSwinForImageModel
    extends ParamsAndFeaturesReadable[SwinForImageClassification]
    with HasPretrained[SwinForImageClassification] {
  override val defaultModelName: Some[String] = Some(
    "image_classifier_swin_base_patch4_window7_224")

  /** Java compliant-overrides */
  override def pretrained(): SwinForImageClassification = super.pretrained()

  override def pretrained(name: String): SwinForImageClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): SwinForImageClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): SwinForImageClassification = super.pretrained(name, lang, remoteLoc)
}

trait ReadSwinForImageDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[SwinForImageClassification] =>

  override val tfFile: String = "image_classification_swin_tensorflow"
  override val onnxFile: String = "image_classification_swin_onnx"

  def readModel(instance: SwinForImageClassification, path: String, spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = instance.getDoNormalize,
      do_resize = instance.getDoRescale,
      feature_extractor_type = "SwinFeatureExtractor",
      image_mean = instance.getImageMean,
      image_std = instance.getImageStd,
      resample = instance.getResample,
      do_rescale = instance.getDoRescale,
      rescale_factor = instance.getRescaleFactor,
      size = instance.getSize)

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

  def loadSavedModel(modelPath: String, spark: SparkSession): SwinForImageClassification = {

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
    val annotatorModel = new SwinForImageClassification()
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
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(wrapper), None, preprocessorConfig)
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

/** This is the companion object of [[SwinForImageClassification]]. Please refer to that class for
  * the documentation.
  */
object SwinForImageClassification
    extends ReadablePretrainedSwinForImageModel
    with ReadSwinForImageDLModel
