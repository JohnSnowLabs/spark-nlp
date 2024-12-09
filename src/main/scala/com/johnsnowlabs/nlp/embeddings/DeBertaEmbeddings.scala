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

import com.johnsnowlabs.ml.ai.DeBerta
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{ReadSentencePieceModel, SentencePieceWrapper, WriteSentencePieceModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{loadSentencePieceAsset, modelSanityCheck, notSupportedEngineError}
import com.johnsnowlabs.ml.util.{ModelEngine, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** The DeBERTa model was proposed in
  * [[https://arxiv.org/abs/2006.03654 DeBERTa: Decoding-enhanced BERT with Disentangled Attention]]
  * by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen It is based on Google’s BERT model
  * released in 2018 and Facebook’s RoBERTa model released in 2019.
  *
  * This model requires input tokenization with SentencePiece model, which is provided by Spark
  * NLP (See tokenizers package).
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = DeBertaEmbeddings.pretrained()
  *  .setInputCols("sentence", "token")
  *  .setOutputCol("embeddings")
  * }}}
  * The default model is `"deberta_v3_base"`, if no name is provided.
  *
  * For extended examples see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/DeBertaEmbeddingsTestSpec.scala DeBertaEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * It builds on RoBERTa with disentangled attention and enhanced mask decoder training with half
  * of the data used in RoBERTa.
  *
  * '''References:'''
  *
  * [[https://github.com/microsoft/DeBERTa]]
  *
  * [[https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/]]
  *
  * '''Paper abstract:'''
  *
  * ''Recent progress in pre-trained neural language models has significantly improved the
  * performance of many natural language processing (NLP) tasks. In this paper we propose a new
  * model architecture DeBERTa (Decoding-enhanced BERT with disentangled attention) that improves
  * the BERT and RoBERTa models using two novel techniques. The first is the disentangled
  * attention mechanism, where each word is represented using two vectors that encode its content
  * and position, respectively, and the attention weights among words are computed using
  * disentangled matrices on their contents and relative positions. Second, an enhanced mask
  * decoder is used to replace the output softmax layer to predict the masked tokens for model
  * pretraining. We show that these two techniques significantly improve the efficiency of model
  * pretraining and performance of downstream tasks. Compared to RoBERTa-Large, a DeBERTa model
  * trained on half of the training data performs consistently better on a wide range of NLP
  * tasks, achieving improvements on MNLI by +0.9% (90.2% vs. 91.1%), on SQuAD v2.0 by +2.3%
  * (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). The DeBERTa code and pre-trained models
  * will be made publicly available at https://github.com/microsoft/DeBERTa.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.DeBertaEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val embeddings = DeBertaEmbeddings.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[1.1342473030090332,-1.3855540752410889,0.9818322062492371,-0.784737348556518...|
  * |[0.847029983997345,-1.047153353691101,-0.1520637571811676,-0.6245765686035156...|
  * |[-0.009860038757324219,-0.13450059294700623,2.707749128341675,1.2916892766952...|
  * |[-0.04192575812339783,-0.5764210224151611,-0.3196685314178467,-0.527840495109...|
  * |[0.15583214163780212,-0.1614152491092682,-0.28423872590065,-0.135491415858268...|
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
class DeBertaEmbeddings(override val uid: String)
    extends AnnotatorModel[DeBertaEmbeddings]
    with HasBatchedAnnotate[DeBertaEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("DEBERTA_EMBEDDINGS"))

  /** Input Annotator Types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Output Annotator Types: WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): DeBertaEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "DeBERTa models do not support sequences longer than 512 because of trainable positional embeddings")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** @group setParam */
  override def setDimension(value: Int): this.type = {
    set(this.dimension, value)
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

  private var _model: Option[Broadcast[DeBerta]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper],
      spp: SentencePieceWrapper): DeBertaEmbeddings = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new DeBerta(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            spp,
            batchSize = $(batchSize),
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures)))
    }

    this
  }

  def getModelIfNotSet: DeBerta = _model.get.value

  setDefault(batchSize -> 8, dimension -> 768, maxSentenceLength -> 128, caseSensitive -> true)

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
    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) =>
        TokenizedWithSentence.unpack(annotations).toArray.map(x => (x, i))
      }

    /*Return empty if the real tokens are empty*/
    val sentenceWordEmbeddings = getModelIfNotSet.predict(
      sentencesWithRow.map(_._1),
      $(batchSize),
      $(maxSentenceLength),
      $(caseSensitive))

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowEmbeddings = sentenceWordEmbeddings
        // zip each annotation with its corresponding row index
        .zip(sentencesWithRow)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowEmbeddings.nonEmpty)
        WordpieceEmbeddingsSentence.pack(rowEmbeddings)
      else
        Seq.empty[Annotation]
    })
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_deberta"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          DeBertaEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          DeBertaEmbeddings.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          DeBertaEmbeddings.openvinoFile)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, suffix, DeBertaEmbeddings.sppFile)

  }

}

trait ReadablePretrainedDeBertaModel
    extends ParamsAndFeaturesReadable[DeBertaEmbeddings]
    with HasPretrained[DeBertaEmbeddings] {
  override val defaultModelName: Some[String] = Some("deberta_v3_base")

  /** Java compliant-overrides */
  override def pretrained(): DeBertaEmbeddings = super.pretrained()

  override def pretrained(name: String): DeBertaEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): DeBertaEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): DeBertaEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadDeBertaDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel
    with ReadOnnxModel
    with ReadOpenvinoModel{
  this: ParamsAndFeaturesReadable[DeBertaEmbeddings] =>

  override val tfFile: String = "deberta_tensorflow"
  override val onnxFile: String = "deberta_onnx"
  override val sppFile: String = "deberta_spp"
  override val openvinoFile: String = "deberta_openvino"

  def readModel(instance: DeBertaEmbeddings, path: String, spark: SparkSession): Unit = {
    val spp = readSentencePieceModel(path, spark, "_deberta_spp", sppFile)

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_deberta_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None, spp)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_deberta_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None, spp)


      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_deberta_openvino")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper), spp)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): DeBertaEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "spm.model")

    /*Universal parameters for all engines*/
    val annotatorModel = new DeBertaEmbeddings()

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
          .setModelIfNotSet(spark, Some(tfWrapper), None, None, spModel)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None, spModel)

        case Openvino.name =>
          val ovWrapper: OpenvinoWrapper =
            OpenvinoWrapper.read(
              spark,
              localModelPath,
              zipped = false,
              useBundle = true,
              detectedEngine = detectedEngine)
          annotatorModel
            .setModelIfNotSet(spark, None, None, Some(ovWrapper), spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[DeBertaEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object DeBertaEmbeddings extends ReadablePretrainedDeBertaModel with ReadDeBertaDLModel
