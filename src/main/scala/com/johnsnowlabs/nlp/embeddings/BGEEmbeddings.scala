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

import com.johnsnowlabs.ml.ai.BGE
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{loadTextAsset, modelSanityCheck, notSupportedEngineError}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForQuestionAnswering
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

/** Sentence embeddings using BGE.
  *
  * BGE, or BAAI General Embeddings, a model that can map any text to a low-dimensional dense
  * vector which can be used for tasks like retrieval, classification, clustering, or semantic
  * search.
  *
  * Note that this annotator is only supported for Spark Versions 3.4 and up.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = BGEEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"bge_base"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=BGE Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BGEEmbeddingsTestSpec.scala BGEEmbeddingsTestSpec]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/pdf/2309.07597 C-Pack: Packaged Resources To Advance General Chinese Embedding]]
  *
  * [[https://github.com/FlagOpen/FlagEmbedding BGE Github Repository]]
  *
  * ''' Paper abstract '''
  *
  * ''We introduce C-Pack, a package of resources that significantly advance the field of general
  * Chinese embeddings. C-Pack includes three critical resources. 1) C-MTEB is a comprehensive
  * benchmark for Chinese text embeddings covering 6 tasks and 35 datasets. 2) C-MTP is a massive
  * text embedding dataset curated from labeled and unlabeled Chinese corpora for training
  * embedding models. 3) C-TEM is a family of embedding models covering multiple sizes. Our models
  * outperform all prior Chinese text embeddings on C-MTEB by up to +10% upon the time of the
  * release. We also integrate and optimize the entire suite of training methods for C-TEM. Along
  * with our resources on general Chinese embedding, we release our data and models for English
  * text embeddings. The English models achieve stateof-the-art performance on the MTEB benchmark;
  * meanwhile, our released English data is 2 times larger than the Chinese data. All these
  * resources are made publicly available at https://github.com/FlagOpen/FlagEmbedding.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.BGEEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = BGEEmbeddings.pretrained("bge_base", "en")
  *   .setInputCols("document")
  *   .setOutputCol("bge_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("bge_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("query: how much protein should a female eat",
  * "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day." +
  * But, as you can see from this chart, you'll need to increase that if you're expecting or training for a" +
  * marathon. Check out the chart below to see how much protein you should be eating each day."
  *
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[[8.0190285E-4, -0.005974853, -0.072875895, 0.007944068, 0.026059335, -0.0080...|
  * |[[0.050514214, 0.010061974, -0.04340176, -0.020937217, 0.05170225, 0.01157857...|
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
class BGEEmbeddings(override val uid: String)
    extends AnnotatorModel[BGEEmbeddings]
    with HasBatchedAnnotate[BGEEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
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
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
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
  private var _model: Option[Broadcast[BGE]] = None

  def this() = this(Identifiable.randomUID("BGE_EMBEDDINGS"))

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): BGEEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "BGE models do not support sequences longer than 512 because of trainable positional embeddings.")
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
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): BGEEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new BGE(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
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

  setDefault(dimension -> 768, batchSize -> 8, maxSentenceLength -> 512, caseSensitive -> false)

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
  def getModelIfNotSet: BGE = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_bge"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          BGEEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          BGEEmbeddings.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          BGEEmbeddings.openvinoFile)

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

trait ReadablePretrainedBGEModel
    extends ParamsAndFeaturesReadable[BGEEmbeddings]
    with HasPretrained[BGEEmbeddings] {
  override val defaultModelName: Some[String] = Some("bge_base")

  /** Java compliant-overrides */
  override def pretrained(): BGEEmbeddings = super.pretrained()

  override def pretrained(name: String): BGEEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BGEEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BGEEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBGEDLModel extends ReadTensorflowModel with ReadOnnxModel  with ReadOpenvinoModel{
  this: ParamsAndFeaturesReadable[BGEEmbeddings] =>

  override val tfFile: String = "bge_tensorflow"
  override val onnxFile: String = "bge_onnx"
  override val openvinoFile: String = "bge_openvino"

  def readModel(instance: BGEEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_bge_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_bge_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_bge_openvino")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BGEEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new BGEEmbeddings()
      .setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(
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
          .setModelIfNotSet(spark, Some(wrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper))



      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[BGEEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object BGEEmbeddings extends ReadablePretrainedBGEModel with ReadBGEDLModel {
  private[BGEEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("BGEEmbeddings")
}
