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
package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.gguf.GGUFWrapperMultiModal
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{LlamaException, LlamaModel}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Multimodal annotator that uses the llama.cpp library to generate text completions with large
  * language models. It supports ingesting images for captioning.
  *
  * At the moment only CLIP based models are supported.
  *
  * For settable parameters, and their explanations, see [[HasLlamaCppInferenceProperties]],
  * [[HasLlamaCppModelProperties]] and refer to the llama.cpp documentation of
  * [[https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server server.cpp]]
  * for more information.
  *
  * If the parameters are not set, the annotator will default to use the parameters provided by
  * the model.
  *
  * This annotator expects a column of annotator type [[AnnotationImage]] for the image and
  * [[Annotation]] for the caption. Note that the image bytes in the image annotation need to be
  * raw image bytes without preprocessing. We provide the helper function
  * [[ImageAssembler.loadImagesAsBytes]] to load the image bytes from a directory.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val autoGGUFVisionModel = AutoGGUFVisionModel.pretrained()
  *   .setInputCols("image', "document")
  *   .setOutputCol("completions")
  * }}}
  * The default model is `"Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFVisionModelTest.scala AutoGGUFVisionModelTest]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFVisionModel.ipynb example notebook]].
  *
  * ==Note==
  * To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
  * the number of GPU layers with the `setNGpuLayers` method.
  *
  * When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
  * according to your hardware to avoid out-of-memory errors.
  *
  * ==Example==
  *
  * {{{
  * import com.johnsnowlabs.nlp.ImageAssembler
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.base._
  * import org.apache.spark.ml.Pipeline
  * import org.apache.spark.sql.DataFrame
  * import org.apache.spark.sql.functions.lit
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("caption")
  *   .setOutputCol("caption_document")
  *
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val imagesPath = "src/test/resources/image/"
  * val data: DataFrame = ImageAssembler
  *   .loadImagesAsBytes(ResourceHelper.spark, imagesPath)
  *   .withColumn("caption", lit("Caption this image.")) // Add a caption to each image.
  *
  * val nPredict = 40
  * val model = AutoGGUFVisionModel.pretrained()
  *   .setInputCols("caption_document", "image_assembler")
  *   .setOutputCol("completions")
  *   .setBatchSize(4)
  *   .setNGpuLayers(99)
  *   .setNCtx(4096)
  *   .setMinKeep(0)
  *   .setMinP(0.05f)
  *   .setNPredict(nPredict)
  *   .setNProbs(0)
  *   .setPenalizeNl(false)
  *   .setRepeatLastN(256)
  *   .setRepeatPenalty(1.18f)
  *   .setStopStrings(Array("</s>", "Llama:", "User:"))
  *   .setTemperature(0.05f)
  *   .setTfsZ(1)
  *   .setTypicalP(1)
  *   .setTopK(40)
  *   .setTopP(0.95f)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, imageAssembler, model))
  * pipeline
  *   .fit(data)
  *   .transform(data)
  *   .selectExpr("reverse(split(image.origin, '/'))[0] as image_name", "completions.result")
  *   .show(truncate = false)
  * +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |image_name       |result                                                                                                                                                                                        |
  * +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
  * |palace.JPEG      |[ The image depicts a large, ornate room with high ceilings and beautifully decorated walls. There are several chairs placed throughout the space, some of which have cushions]               |
  * |egyptian_cat.jpeg|[ The image features two cats lying on a pink surface, possibly a bed or sofa. One cat is positioned towards the left side of the scene and appears to be sleeping while holding]             |
  * |hippopotamus.JPEG|[ A large brown hippo is swimming in a body of water, possibly an aquarium. The hippo appears to be enjoying its time in the water and seems relaxed as it floats]                            |
  * |hen.JPEG         |[ The image features a large chicken standing next to several baby chickens. In total, there are five birds in the scene: one adult and four young ones. They appear to be gathered together] |
  * |ostrich.JPEG     |[ The image features a large, long-necked bird standing in the grass. It appears to be an ostrich or similar species with its head held high and looking around. In addition to]              |
  * |junco.JPEG       |[ A small bird with a black head and white chest is standing on the snow. It appears to be looking at something, possibly food or another animal in its vicinity. The scene takes place out]  |
  * |bluetick.jpg     |[ A dog with a red collar is sitting on the floor, looking at something. The dog appears to be staring into the distance or focusing its attention on an object in front of it.]              |
  * |chihuahua.jpg    |[ A small brown dog wearing a sweater is sitting on the floor. The dog appears to be looking at something, possibly its owner or another animal in the room. It seems comfortable and relaxed]|
  * |tractor.JPEG     |[ A man is sitting in the driver's seat of a green tractor, which has yellow wheels and tires. The tractor appears to be parked on top of an empty field with]                                |
  * |ox.JPEG          |[ A large bull with horns is standing in a grassy field.]                                                                                                                                     |
  * +-----------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
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
class AutoGGUFVisionModel(override val uid: String)
    extends AnnotatorModel[AutoGGUFVisionModel]
    with HasBatchedAnnotateTextImage[AutoGGUFVisionModel]
    with HasEngine
    with HasLlamaCppModelProperties
    with HasLlamaCppInferenceProperties
    with HasProtectedParams
    with CompletionPostProcessing {
  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(AnnotatorType.IMAGE, AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("AutoGGUFVisionModel"))

  private var _model: Option[Broadcast[GGUFWrapperMultiModal]] = None

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapperMultiModal = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapperMultiModal): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(wrapper))
    }

    this
  }

  /** Closes the llama.cpp model backend freeing resources. The model is reloaded when used again.
    */
  def close(): Unit = GGUFWrapperMultiModal.closeBroadcastModel(_model)

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  /** Sets the number of parallel processes for decoding. This is an alias for `setBatchSize`.
    *
    * @group setParam
    * @param nParallel
    *   The number of parallel processes for decoding
    */
  def setNParallel(nParallel: Int): this.type = {
    setBatchSize(nParallel)
  }

  setDefault(
    engine -> LlamaCPP.name,
    useChatTemplate -> true,
    nCtx -> 4096,
    nBatch -> 512,
    nPredict -> 100,
    nGpuLayers -> 99,
    systemPrompt -> "You are a helpful assistant.",
    batchSize -> 2)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getModelIfNotSet.saveToFile(path)
  }

  /** Completes the batch of annotations.
    *
    * @param batchedAnnotations
    *   The single batch of annotations
    * @return
    *   Completed text sequences
    *
    * sentences that belong to the same original row !! (challenging)
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[(Annotation, AnnotationImage)]): Seq[Seq[Annotation]] = {
    if (batchedAnnotations.nonEmpty) {

      // set parallel decoding to batch size
      val modelParams = getModelParameters.setParallel(getBatchSize)
      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

      val (prompts, base64EncodedImages) = batchedAnnotations.unzip match {
        case (promptAnnotations, imageAnnotations) =>
          (
            promptAnnotations.map(_.result).toArray,
            imageAnnotations
              .map(imgAnno => ImageIOUtils.encodeImageBase64(imgAnno.result))
              .toArray)
      }

      val textAndMeta: Array[(String, Map[String, String])] = prompts
        .zip(base64EncodedImages)
        .map { case (prompt, base64Image) =>
          try {
            val results = LlamaExtensions.completeImage(
              model,
              getInferenceParameters,
              getSystemPrompt,
              prompt,
              base64Image)
            val resultsCleaned = processCompletions(Array(results)).head
            (resultsCleaned, Map.empty[String, String])
          } catch {
            case e: LlamaException =>
              logger.error("Error in llama.cpp image batch completion", e)
              ("", Map("llamacpp_exception" -> e.getMessage))
          }
        }

      val result: Seq[Seq[Annotation]] =
        batchedAnnotations.zip(textAndMeta).map {
          case (
                (textAnnotation: Annotation, imageAnnotation: AnnotationImage),
                (text: String, metadata: Map[String, String])) =>
            val totalMetadata =
              textAnnotation.metadata ++ imageAnnotation.metadata ++ metadata
            Seq(new Annotation(outputAnnotatorType, 0, text.length - 1, text, totalMetadata))
        }
      result
    } else Seq(Seq.empty[Annotation])
  }
}

trait ReadablePretrainedAutoGGUFVisionModel
    extends ParamsAndFeaturesFallbackReadable[AutoGGUFVisionModel]
    with HasPretrained[AutoGGUFVisionModel] {
  override val defaultModelName: Some[String] = Some("Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): AutoGGUFVisionModel = super.pretrained()

  override def pretrained(name: String): AutoGGUFVisionModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): AutoGGUFVisionModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AutoGGUFVisionModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAutoGGUFVisionModel {
  this: ParamsAndFeaturesFallbackReadable[AutoGGUFVisionModel] =>

  override def fallbackLoad(folder: String, spark: SparkSession): AutoGGUFVisionModel = {
    val actualFolderPath: String = ResourceHelper.resolvePath(folder)

    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val (ggufFile, mmprojFile) = GGUFWrapperMultiModal.findGGUFModelsInFolder(localFolder)
    loadSavedModel(ggufFile, mmprojFile, spark)
  }

  def readModel(instance: AutoGGUFVisionModel, path: String, spark: SparkSession): Unit = {
    val model: GGUFWrapperMultiModal = GGUFWrapperMultiModal.readModel(path, spark)

    instance.setModelIfNotSet(spark, model)
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      mmprojPath: String,
      spark: SparkSession): AutoGGUFVisionModel = {
    // TODO potentially enable download from HF-URLS
    val localPathModel: String = ResourceHelper.copyToLocal(modelPath)
    val localPathMmproj: String = ResourceHelper.copyToLocal(mmprojPath)

    val annotatorModel = new AutoGGUFVisionModel()
    val wrapper = GGUFWrapperMultiModal.read(spark, localPathModel, localPathMmproj)

    annotatorModel
      .setModelIfNotSet(spark, wrapper)
      .setEngine(LlamaCPP.name)

    // TODO mmproj metadata necessary?
    val metadata = LlamaExtensions.getMetadataFromFile(localPathModel)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFVisionModel]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFVisionModel
    extends ReadablePretrainedAutoGGUFVisionModel
    with ReadAutoGGUFVisionModel
