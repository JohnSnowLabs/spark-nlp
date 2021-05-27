/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.HasStorageRef

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File

/**
 * The DistilBERT model was proposed in the paper '''DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter'''
 * [[https://arxiv.org/abs/1910.01108]].
 * DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than
 * `bert-base-uncased`, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.
 *
 * The abstract from the paper is the following:
 *
 * As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
 * operating these large models in on-the-edge and/or under constrained computational training or inference budgets
 * remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
 * model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
 * counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage
 * knowledge distillation during the pretraining phase and show that it is possible to reduce the size of a BERT model by
 * 40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive
 * biases learned by larger models during pretraining, we introduce a triple loss combining language modeling,
 * distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
 * demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device
 * study.
 *
 * Tips:
 *
 * - DistilBERT doesn't have :obj:`token_type_ids`, you don't need to indicate which token belongs to which segment. Just
 * separate your segments with the separation token :obj:`tokenizer.sep_token` (or :obj:`[SEP]`).
 *
 * - DistilBERT doesn't have options to select the input positions (:obj:`position_ids` input). This could be added if
 * necessary though, just let us know if you need this option.
 *
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
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
 * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 */
class DistilBertEmbeddings(override val uid: String)
  extends AnnotatorModel[DistilBertEmbeddings]
    with HasBatchedAnnotate[DistilBertEmbeddings]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("DISTILBERT_EMBEDDINGS"))

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /**
   * Vocabulary used to encode the words to ids with WordPieceEncoder
   *
   * @group param
   * */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): DistilBertEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "DistilBERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /**
   * It contains TF model signatures for the laded saved model
   *
   * @group param
   * */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowDistilBert]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TensorflowWrapper): DistilBertEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowDistilBert(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures
          )
        )
      )
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowDistilBert = _model.get.value

  /** Set Embeddings dimensions for the DistilBERT model
   * Only possible to set this when the first time is saved
   * dimension is not changeable, it comes from DistilBERT config file
   *
   * @group setParam
   * */
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this
  }

  /** Whether to lowercase tokens or not
   *
   * @group setParam
   * */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(
    dimension -> 768,
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> false
  )

  def tokenizeWithAlignment(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens = tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map { token =>
        val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
        val sentenceBegin = token.begin
        val sentenceEnd = token.end
        val sentenceInedx = tokenIndex.sentenceIndex
        val result = basicTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceInedx))
        if (result.nonEmpty) result.head else IndexedToken("")
      }
      val wordpieceTokens = bertTokens.flatMap(token => encoder.encode(token)).take($(maxSentenceLength))
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  /**
   * takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @param batchedAnnotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
   * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    val batchedTokenizedSentences: Array[Array[TokenizedSentence]] = batchedAnnotations.map(annotations =>
      TokenizedWithSentence.unpack(annotations).toArray
    ).toArray
    /*Return empty if the real tokens are empty*/
    if (batchedTokenizedSentences.nonEmpty) batchedTokenizedSentences.map(tokenizedSentences => {
      val tokenized = tokenizeWithAlignment(tokenizedSentences)

      val withEmbeddings = getModelIfNotSet.calculateEmbeddings(
        tokenized,
        tokenizedSentences,
        $(batchSize),
        $(maxSentenceLength),
        $(caseSensitive)
      )
      WordpieceEmbeddingsSentence.pack(withEmbeddings)
    }) else {
      Seq(Seq.empty[Annotation])
    }
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflowWrapper, "_distilbert", DistilBertEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedDistilBertModel extends ParamsAndFeaturesReadable[DistilBertEmbeddings] with HasPretrained[DistilBertEmbeddings] {
  override val defaultModelName: Some[String] = Some("distilbert_base_cased")

  /** Java compliant-overrides */
  override def pretrained(): DistilBertEmbeddings = super.pretrained()

  override def pretrained(name: String): DistilBertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): DistilBertEmbeddings = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): DistilBertEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadDistilBertTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[DistilBertEmbeddings] =>

  override val tfFile: String = "distilbert_tensorflow"

  def readTensorflow(instance: DistilBertEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_distilbert_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): DistilBertEmbeddings = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath"
    )

    val vocab = new File(tfModelPath + "/assets", "vocab.txt")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $tfModelPath")

    val vocabResource = new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val (wrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important is we use getSignatures inside setModelIfNotSet */
    new DistilBertEmbeddings()
      .setVocabulary(words)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}


object DistilBertEmbeddings extends ReadablePretrainedDistilBertModel with ReadDistilBertTensorflowModel
