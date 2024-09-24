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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.Instructor
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

/** Sentence embeddings using INSTRUCTOR.
  *
  * Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text
  * embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation,
  * etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction,
  * without any finetuning. Instructor👨‍ achieves sota on 70 diverse embedding tasks!
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = InstructorEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("instructor_embeddings")
  * }}}
  * The default model is `"instructor_base"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=Instructor Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/InstructorEmbeddingsTestSpec.scala InstructorEmbeddingsTestSpec]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/2212.09741 One Embedder, Any Task: Instruction-Finetuned Text Embeddings]]
  *
  * [[https://github.com/HKUNLP/instructor-embedding/ INSTRUCTOR Github Repository]]
  *
  * ''' Paper abstract '''
  *
  * ''We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions:
  * every text input is embedded together with instructions explaining the use case (e.g., task
  * and domain descriptions). Unlike encoders from prior work that are more specialized,
  * INSTRUCTOR is a single embedder that can generate text embeddings tailored to different
  * downstream tasks and domains, without any further training. We first annotate instructions for
  * 330 diverse tasks and train INSTRUCTOR on this multitask mixture with a contrastive loss. We
  * evaluate INSTRUCTOR on 70 embedding evaluation tasks (66 of which are unseen during training),
  * ranging from classification and information retrieval to semantic textual similarity and text
  * generation evaluation. INSTRUCTOR, while having an order of magnitude fewer parameters than
  * the previous best model, achieves state-of-the-art performance, with an average improvement of
  * 3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests
  * that INSTRUCTOR is robust to changes in instructions, and that instruction finetuning
  * mitigates the challenge of training a single model on diverse datasets. Our model, code, and
  * data are available at this https URL. [[https://instructor-embedding.github.io/]] ''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.InstructorEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = InstructorEmbeddings.pretrained("instructor_base", "en")
  *   .setInputCols("document")
  *   .setInstruction("Represent the Medicine sentence for clustering: ")
  *   .setOutputCol("instructor_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("instructor_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based embeddings
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
class InstructorEmbeddings(override val uid: String)
    extends AnnotatorModel[InstructorEmbeddings]
    with HasBatchedAnnotate[InstructorEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with WriteSentencePieceModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** Set transformer instruction, e.g. 'summarize' format: `"instruction:"`.
    *
    * @group param
    */
  val instruction =
    new Param[String](this, "instruction", "Set transformer instruction, e.g. 'summarize'")

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()
  private var _model: Option[Broadcast[Instructor]] = None

  def this() = this(Identifiable.randomUID("INSTRUCTOR_EMBEDDINGS"))

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): InstructorEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "Instructor models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  def setInstruction(value: String): InstructorEmbeddings.this.type = {
    set(instruction, value)
    this
  }

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      spp: SentencePieceWrapper): InstructorEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Instructor(
            tensorflowWrapper,
            onnxWrapper,
            spp = spp,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures)))
    }

    this
  }

  /** Set Embeddings dimensions for the BERT model Only possible to set this when the first time
    * is saved dimension is not changeable, it comes from BERT config file
    *
    * @group setParam
    */
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this
  }

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(
    dimension -> 768,
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> false,
    instruction -> "")

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }
    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        batchSize = $(batchSize),
        maxSentenceLength = $(maxSentenceLength),
        instruction = $(instruction))
    } else {
      Seq()
    }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = processedAnnotations
        // zip each annotation with its corresponding row index
        .zip(allAnnotations)
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

  /** @group getParam */
  def getModelIfNotSet: Instructor = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {

    super.onWrite(path, spark)
    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          "_instructor",
          InstructorEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)

      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_instructor",
          InstructorEmbeddings.onnxFile)
    }
    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      "_instructor",
      InstructorEmbeddings.sppFile)

  }

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }

}

trait ReadablePretrainedInstructorModel
    extends ParamsAndFeaturesReadable[InstructorEmbeddings]
    with HasPretrained[InstructorEmbeddings] {
  override val defaultModelName: Some[String] = Some("instructor_base")

  /** Java compliant-overrides */
  override def pretrained(): InstructorEmbeddings = super.pretrained()

  override def pretrained(name: String): InstructorEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): InstructorEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): InstructorEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadInstructorDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel
    with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[InstructorEmbeddings] =>

  override val tfFile: String = "instructor_tensorflow"
  override val sppFile: String = "instructor_spp"
  override val onnxFile: String = "instructor_onnx"

  def readModel(instance: InstructorEmbeddings, path: String, spark: SparkSession): Unit = {
    val spp = readSentencePieceModel(path, spark, "_instructor_spp", sppFile)

    instance.getEngine match {
      case TensorFlow.name =>
        val tf = readTensorflowModel(
          path,
          spark,
          "_instructor_tf",
          savedSignatures = instance.getSignatures,
          initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tf), None, spp)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_instructor_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), spp)

    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): InstructorEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    /*Universal parameters for all engines*/
    val annotatorModel = new InstructorEmbeddings()

    annotatorModel.set(annotatorModel.engine, detectedEngine)
    val spModel = loadSentencePieceAsset(localModelPath, "spiece.model")
    detectedEngine match {
      case TensorFlow.name =>
        val (tfwrapper, signatures) = TensorflowWrapper.read(
          localModelPath,
          zipped = false,
          useBundle = true,
          tags = Array("serve"),
          initAllTables = false)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(tfwrapper), None, spModel)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), spModel)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[InstructorEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object InstructorEmbeddings
    extends ReadablePretrainedInstructorModel
    with ReadInstructorDLModel
    with ReadSentencePieceModel {
  private[InstructorEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("InstructorEmbeddings")
}
