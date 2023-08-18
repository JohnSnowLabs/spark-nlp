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

import com.johnsnowlabs.ml.ai.Bert
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import com.johnsnowlabs.util.FileHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import java.nio.file.Files
import java.util.UUID

/** Token-level embeddings using BERT. BERT (Bidirectional Encoder Representations from
  * Transformers) provides dense vector representations for natural language by using a deep,
  * pre-trained neural network with the Transformer architecture.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = BertEmbeddings.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("bert_embeddings")
  * }}}
  * The default model is `"small_bert_L2_768"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_bert.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala BertEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/1810.04805 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]]
  *
  * [[https://github.com/google-research/bert]]
  *
  * ''' Paper abstract '''
  *
  * ''We introduce a new language representation model called BERT, which stands for Bidirectional
  * Encoder Representations from Transformers. Unlike recent language representation models, BERT
  * is designed to pre-train deep bidirectional representations from unlabeled text by jointly
  * conditioning on both left and right context in all layers. As a result, the pre-trained BERT
  * model can be fine-tuned with just one additional output layer to create state-of-the-art
  * models for a wide range of tasks, such as question answering and language inference, without
  * substantial task-specific architecture modifications. BERT is conceptually simple and
  * empirically powerful. It obtains new state-of-the-art results on eleven natural language
  * processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement),
  * MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1
  * to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute
  * improvement).''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.BertEmbeddings
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
  * val embeddings = BertEmbeddings.pretrained("small_bert_L2_128", "en")
  *   .setInputCols("token", "document")
  *   .setOutputCol("bert_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("bert_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
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
  * |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
  * |[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
  * |[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
  * |[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
  * |[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[BertSentenceEmbeddings]] for sentence-level embeddings
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.BertForTokenClassification BertForTokenClassification]]
  *   For BertEmbeddings with a token classification layer on top
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
class BertEmbeddings(override val uid: String)
    extends AnnotatorModel[BertEmbeddings]
    with HasBatchedAnnotate[BertEmbeddings]
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
  def this() = this(Identifiable.randomUID("BERT_EMBEDDINGS"))

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

  /** @group setParam */
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

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): BertEmbeddings.this.type =
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
      "BERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

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

  private var _model: Option[Broadcast[Bert]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): BertEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Bert(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.wordEmbeddings)))
    }

    this
  }

  def getModelIfNotSet: Bert = _model.get.value

  /** Set Embeddings dimensions for the BERT model Only possible to set this when the first time
    * is saved dimension is not changeable, it comes from BERT config file
    *
    * @group setParam
    */
  override def setDimension(value: Int): this.type = {
    set(this.dimension, value)
  }

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    set(this.caseSensitive, value)
  }

  setDefault(dimension -> 768, batchSize -> 8, maxSentenceLength -> 128, caseSensitive -> false)

  def tokenizeWithAlignment(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result = basicTokenizer.tokenize(
              Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens =
        bertTokens.flatMap(token => encoder.encode(token)).take($(maxSentenceLength))
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

    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) =>
        TokenizedWithSentence.unpack(annotations).toArray.map(x => (x, i))
      }

    // Tokenize sentences
    val tokenizedSentences = tokenizeWithAlignment(sentencesWithRow.map(_._1))

    // Process all sentences
    val sentenceWordEmbeddings = getModelIfNotSet.predict(
      tokenizedSentences,
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
    val suffix = "_bert"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          BertEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          BertEmbeddings.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          suffix,
          BertEmbeddings.openvinoFile)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedBertModel
    extends ParamsAndFeaturesReadable[BertEmbeddings]
    with HasPretrained[BertEmbeddings] {
  override val defaultModelName: Some[String] = Some("small_bert_L2_768")

  /** Java compliant-overrides */
  override def pretrained(): BertEmbeddings = super.pretrained()

  override def pretrained(name: String): BertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BertEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BertEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBertDLModel extends ReadTensorflowModel with ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[BertEmbeddings] =>

  override val tfFile: String = "bert_tensorflow"
  override val onnxFile: String = "bert_onnx"
  override val openvinoFile: String = "bert_openvino"

  def readModel(instance: BertEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_bert_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_bert_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)
      }

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_bert_openvino")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): BertEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new BertEmbeddings()
      .setVocabulary(vocabs)
    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    modelEngine match {
      case Openvino.name =>
        val tmpFolder = Files
          .createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_ov_model")
          .toAbsolutePath
          .toString

        /** Convert the model from the detected framework to Openvino Intermediate format */
        val irModelFolder =
          if (detectedEngine == Openvino.name) {
            localModelPath
          } else {
            OpenvinoWrapper.convertToOpenvinoFormat(
              modelPath = localModelPath,
              targetPath = tmpFolder,
              detectedEngine = detectedEngine,
              zipped = false,
              useBundle = true)
            tmpFolder
          }
        val (ovWrapper: OpenvinoWrapper, tensorNames: Map[String, String]) =
          OpenvinoWrapper.fromOpenvinoFormat(irModelFolder, zipped = false)

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel.setSignatures(tensorNames)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper))
        FileHelper.delete(tmpFolder)

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
          .setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper = OnnxWrapper.read(localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

/** This is the companion object of [[BertEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object BertEmbeddings extends ReadablePretrainedBertModel with ReadBertDLModel {
  private[BertEmbeddings] val logger: Logger = LoggerFactory.getLogger("BertEmbeddings")
}
