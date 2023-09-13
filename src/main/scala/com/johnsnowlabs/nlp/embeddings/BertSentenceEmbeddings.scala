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
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Sentence-level embeddings using BERT. BERT (Bidirectional Encoder Representations from
  * Transformers) provides dense vector representations for natural language by using a deep,
  * pre-trained neural network with the Transformer architecture.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = BertSentenceEmbeddings.pretrained()
  *   .setInputCols("sentence")
  *   .setOutputCol("sentence_bert_embeddings")
  * }}}
  * The default model is `"sent_small_bert_L2_768"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddingsTestSpec.scala BertSentenceEmbeddingsTestSpec]].
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
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128")
  *   .setInputCols("sentence")
  *   .setOutputCol("sentence_bert_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("sentence_bert_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("John loves apples. Mary loves oranges. John loves Mary.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[-0.8951074481010437,0.13753940165042877,0.3108254075050354,-1.65693199634552...|
  * |[-0.6180210709571838,-0.12179657071828842,-0.191165953874588,-1.4497021436691...|
  * |[-0.822715163230896,0.7568016648292542,-0.1165061742067337,-1.59048593044281,...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[BertSentenceEmbeddings]] for sentence-level embeddings
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.BertForSequenceClassification BertForSequenceClassification]]
  *   for embeddings with a sequence classification layer on top
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
class BertSentenceEmbeddings(override val uid: String)
    extends AnnotatorModel[BertSentenceEmbeddings]
    with HasBatchedAnnotate[BertSentenceEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine
    with HasProtectedParams {

  def this() = this(Identifiable.randomUID("BERT_SENTENCE_EMBEDDINGS"))

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
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

  /** Use Long type instead of Int type for inputs (Default: `false`)
    *
    * @group param
    */
  val isLong = new BooleanParam(
    parent = this,
    name = "isLong",
    "Use Long type instead of Int type for inputs buffer - Some Bert models require Long instead of Int.")
    .setProtected()

  /** set isLong
    *
    * @group setParam
    */
  def setIsLong(value: Boolean): this.type = {
    set(this.isLong, value)
  }

  /** get isLong
    *
    * @group getParam
    */
  def getIsLong: Boolean = $(isLong)

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

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

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group setParam
    */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): BertSentenceEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** Max sentence length to process (Default: `128`)
    *
    * @group setParam
    */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "BERT models do not support sequences longer than 512 because of trainable positional embeddings")

    set(maxSentenceLength, value)
  }

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
    *
    * @group getParam
    */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  setDefault(
    dimension -> 768,
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> false,
    isLong -> false)

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

  /** @group getParam */
  def getModelIfNotSet: Bert = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new Bert(
            tensorflowWrapper,
            onnxWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.sentenceEmbeddings,
            isSBert = getIsLong)))
    }

    this
  }

  def tokenize(sentences: Seq[Sentence]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))
    sentences.map { s =>
      val tokens = basicTokenizer.tokenize(s)
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

    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) => SentenceSplit.unpack(annotations).map(x => (x, i)) }

    // Tokenize sentences
    val tokenizedSentences = tokenize(sentencesWithRow.map(_._1))

    // Process all sentences
    val allAnnotations = getModelIfNotSet.predictSequence(
      tokenizedSentences,
      sentencesWithRow.map(_._1),
      $(batchSize),
      $(maxSentenceLength),
      getIsLong,
      sparkSession)

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = allAnnotations
        // zip each annotation with its corresponding row index
        .zip(sentencesWithRow)
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

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          "_bert_sentence",
          BertSentenceEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_bert_sentence",
          BertSentenceEmbeddings.onnxFile)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

}

trait ReadablePretrainedBertSentenceModel
    extends ParamsAndFeaturesReadable[BertSentenceEmbeddings]
    with HasPretrained[BertSentenceEmbeddings] {
  override val defaultModelName: Some[String] = Some("sent_small_bert_L2_768")

  /** Java compliant-overrides */
  override def pretrained(): BertSentenceEmbeddings = super.pretrained()

  override def pretrained(name: String): BertSentenceEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BertSentenceEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BertSentenceEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBertSentenceDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[BertSentenceEmbeddings] =>

  override val tfFile: String = "bert_sentence_tensorflow"
  override val onnxFile: String = "bert_sentence_onnx"

  def readModel(instance: BertSentenceEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper =
          readTensorflowModel(path, spark, "_bert_sentence_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name => {
        val onnxWrapper =
          readOnnxModel(path, spark, "_bert_sentence_onnx", zipped = true, useBundle = false)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
      }
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BertSentenceEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new BertSentenceEmbeddings()
      .setVocabulary(vocabs)

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
          .setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapper = OnnxWrapper.read(
          localModelPath,
          zipped = false,
          useBundle = true,
          sparkSession = Some(spark))
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[BertSentenceEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object BertSentenceEmbeddings
    extends ReadablePretrainedBertSentenceModel
    with ReadBertSentenceDLModel
