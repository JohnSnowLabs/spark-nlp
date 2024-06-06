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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.MPNet
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

/** Sentence embeddings using MPNet.
  *
  * The MPNet model was proposed in MPNet: Masked and Permuted Pre-training for Language
  * Understanding by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu. MPNet adopts a novel
  * pre-training method, named masked and permuted language modeling, to inherit the advantages of
  * masked language modeling and permuted language modeling for natural language understanding.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = MPNetEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("mpnet_embeddings")
  * }}}
  * The default model is `"all_mpnet_base_v2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=MPNet Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/MPNetEmbeddingsTestSpec.scala MPNetEmbeddingsTestSpec]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/2004.09297 MPNet: Masked and Permuted Pre-training for Language Understanding]]
  *
  * [[https://github.com/microsoft/MPNet MPNet Github Repository]]
  *
  * ''' Paper abstract '''
  *
  * ''BERT adopts masked language modeling (MLM) for pre-training and is one of the most
  * successful pre-training models. Since BERT neglects dependency among predicted tokens, XLNet
  * introduces permuted language modeling (PLM) for pre-training to address this problem. However,
  * XLNet does not leverage the full position information of a sentence and thus suffers from
  * position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a
  * novel pre-training method that inherits the advantages of BERT and XLNet and avoids their
  * limitations. MPNet leverages the dependency among predicted tokens through permuted language
  * modeling (vs. MLM in BERT), and takes auxiliary position information as input to make the
  * model see a full sentence and thus reducing the position discrepancy (vs. PLM in XLNet). We
  * pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety
  * of down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms
  * MLM and PLM by a large margin, and achieves better results on these tasks compared with
  * previous state-of-the-art pre-trained methods (e.g., BERT, XLNet, RoBERTa) under the same
  * model setting.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = MPNetEmbeddings.pretrained("all_mpnet_base_v2", "en")
  *   .setInputCols("document")
  *   .setOutputCol("mpnet_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("mpnet_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("This is an example sentence", "Each sentence is converted").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[[0.022502584, -0.078291744, -0.023030775, -0.0051000593, -0.080340415, 0.039...|
  * |[[0.041702367, 0.0010974605, -0.015534201, 0.07092203, -0.0017729357, 0.04661...|
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
class MPNetEmbeddings(override val uid: String)
    extends AnnotatorModel[MPNetEmbeddings]
    with HasBatchedAnnotate[MPNetEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEmbeddingsProperties
    with HasStorageRef
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

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()
  private var _model: Option[Broadcast[MPNet]] = None

  def this() = this(Identifiable.randomUID("MPNET_EMBEDDINGS"))

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): MPNetEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "MPNet models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

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
      onnxWrapper: Option[OnnxWrapper]): MPNetEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MPNet(
            tensorflowWrapper,
            onnxWrapper,
            configProtoBytes = getConfigProtoBytes,
            sentenceStartTokenId = sentenceStartTokenId,
            sentenceEndTokenId = sentenceEndTokenId,
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

  setDefault(dimension -> 768, batchSize -> 8, maxSentenceLength -> 128, caseSensitive -> false)

  def tokenize(sentences: Seq[Annotation]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))
    sentences.map { s =>
      val sent = Sentence(
        content = s.result,
        start = s.begin,
        end = s.end,
        metadata = Some(s.metadata),
        index = s.begin)
      val tokens = basicTokenizer.tokenize(sent)
      val wordpieceTokens = tokens.flatMap(token => encoder.encode(token))
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

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

    // Tokenize sentences
    val tokenizedSentences = tokenize(allAnnotations.map(_._1))
    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        tokenizedSentences = tokenizedSentences,
        batchSize = $(batchSize),
        maxSentenceLength = $(maxSentenceLength))
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
  def getModelIfNotSet: MPNet = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_mpnet"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          MPNetEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          MPNetEmbeddings.onnxFile)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
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

trait ReadablePretrainedMPNetModel
    extends ParamsAndFeaturesReadable[MPNetEmbeddings]
    with HasPretrained[MPNetEmbeddings] {
  override val defaultModelName: Some[String] = Some("all_mpnet_base_v2")

  /** Java compliant-overrides */
  override def pretrained(): MPNetEmbeddings = super.pretrained()

  override def pretrained(name: String): MPNetEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): MPNetEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MPNetEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMPNetDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[MPNetEmbeddings] =>

  override val tfFile: String = "mpnet_tensorflow"
  override val onnxFile: String = "mpnet_onnx"
  def readModel(instance: MPNetEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_mpnet_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_mpnet_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MPNetEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)
    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    /*Universal parameters for all engines*/
    val annotatorModel = new MPNetEmbeddings().setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)
    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) = TensorflowWrapper.read(
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
          .setModelIfNotSet(spark, Some(wrapper), None)

      case ONNX.name =>
        val onnxWrapper = OnnxWrapper.read(localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MPNetEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object MPNetEmbeddings extends ReadablePretrainedMPNetModel with ReadMPNetDLModel {
  private[MPNetEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("MPNetEmbeddings")
}
