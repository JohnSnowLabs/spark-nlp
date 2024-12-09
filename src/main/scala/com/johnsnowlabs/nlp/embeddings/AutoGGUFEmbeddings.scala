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

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.llama.LlamaModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** Annotator that uses the llama.cpp library to generate text embeddings with large language
  * models.
  *
  * The type of embedding pooling can be set with the `setPoolingType` method. The default is
  * `"MEAN"`. The available options are `"NONE"`, `"MEAN"`, `"CLS"`, and `"LAST"`.
  *
  * For all settable parameters, and their explanations, see [[HasLlamaCppModelProperties]].
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val autoGGUFModel = AutoGGUFEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"nomic-embed-text-v1.5.Q8_0.gguf"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/AutoGGUFEmbeddingsTest.scala AutoGGUFEmbeddingsTest]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFEmbeddings.ipynb example notebook]].
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
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  * import spark.implicits._
  *
  * val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  *
  * val autoGGUFModel = AutoGGUFEmbeddings
  *   .pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("embeddings")
  *   .setBatchSize(4)
  *   .setPoolingType("MEAN")
  *
  * val pipeline = new Pipeline().setStages(Array(document, autoGGUFModel))
  *
  * val data = Seq(
  *   "The moons of Jupiter are 77 in total, with 79 confirmed natural satellites and 2 man-made ones.")
  *   .toDF("text")
  * val result = pipeline.fit(data).transform(data)
  * result.select("embeddings.embeddings").show(truncate = false)
  * +--------------------------------------------------------------------------------+
  * |                                                                      embeddings|
  * +--------------------------------------------------------------------------------+
  * |[[-0.034486726, 0.07770534, -0.15982522, -0.017873349, 0.013914132, 0.0365736...|
  * +--------------------------------------------------------------------------------+
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
class AutoGGUFEmbeddings(override val uid: String)
    extends AnnotatorModel[AutoGGUFEmbeddings]
    with HasBatchedAnnotate[AutoGGUFEmbeddings]
    with HasEngine
    with HasLlamaCppModelProperties
    with HasProtectedParams {

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("AutoGGUFModel"))

  private var _model: Option[Broadcast[GGUFWrapper]] = None

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(wrapper))
    }

    setGpuSupportIfAvailable(spark)
  }

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  setDefault(
    engine -> LlamaCPP.name,
    embedding -> true,
    poolingType -> "MEAN",
    nCtx -> 4096,
    nBatch -> 512)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getModelIfNotSet.saveToFile(path)
  }

  /** Completes the batch of annotations.
    *
    * @param batchedAnnotations
    *   Annotations (single element arrays) in batches
    * @return
    *   Completed text sequences
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    require(
      getEmbedding,
      "Embeddings have been manually disabled. Please enable them with setEmbedding(true).")
    val annotations: Seq[Annotation] = batchedAnnotations.flatten
    if (annotations.nonEmpty) {

      val modelParams =
        getModelParameters.setNParallel(getBatchSize) // set parallel decoding to batch size

      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)

      val annotationsText = annotations.map(_.result)

      // Return embeddings in annotation
      val (embeddings: Array[Array[Float]], metadata: Map[String, String]) =
        try {
          (model.requestBatchEmbeddings(annotationsText.toArray), Map.empty)
        } catch {
          case e: Exception =>
            logger.error("Error in llama.cpp embeddings", e)
            (Array.empty[Array[Float]], Map("llamacpp_exception" -> e.getMessage))
        }

      // Choose empty text for result annotations
      annotations.zip(embeddings).map { case (annotation, embedding) =>
        Seq(
          new Annotation(
            annotatorType = annotation.annotatorType,
            begin = annotation.begin,
            end = annotation.end,
            result = annotation.result,
            metadata = annotation.metadata ++ metadata,
            embeddings = embedding))
      }
    } else Seq(Seq.empty[Annotation])
  }
}

trait ReadablePretrainedAutoGGUFEmbeddings
    extends ParamsAndFeaturesReadable[AutoGGUFEmbeddings]
    with HasPretrained[AutoGGUFEmbeddings] {
  override val defaultModelName: Some[String] = Some("nomic-embed-text-v1.5.Q8_0.gguf")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): AutoGGUFEmbeddings = super.pretrained()

  override def pretrained(name: String): AutoGGUFEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): AutoGGUFEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AutoGGUFEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAutoGGUFEmbeddings {
  this: ParamsAndFeaturesReadable[AutoGGUFEmbeddings] =>

  def readModel(instance: AutoGGUFEmbeddings, path: String, spark: SparkSession): Unit = {
    val model: GGUFWrapper = GGUFWrapper.readModel(path, spark)
    instance.setModelIfNotSet(spark, model)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AutoGGUFEmbeddings = {
    // TODO potentially enable download from HF-URLS
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val annotatorModel = new AutoGGUFEmbeddings()
    annotatorModel
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))
      .setEngine(LlamaCPP.name)

    val metadata = LlamaModel.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFEmbeddings extends ReadablePretrainedAutoGGUFEmbeddings with ReadAutoGGUFEmbeddings
