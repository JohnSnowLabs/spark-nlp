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
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BpeTokenizer
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.HasStorageRef

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File

/**
 * The XLM model was proposed in '''Cross-lingual Language Model Pretraining''' [[https://arxiv.org/abs/1901.07291]] by
 * Guillaume Lample, Alexis Conneau. It's a transformer pretrained using one of the following objectives:
 *
 * - a causal language modeling (CLM) objective (next token prediction),
 * - a masked language modeling (MLM) objective (BERT-like), or
 * - a Translation Language Modeling (TLM) object (extension of BERT's MLM to multiple language inputs)
 *
 * The abstract from the paper is the following:
 *
 * Recent studies have demonstrated the efficiency of generative pretraining for English natural language understanding.
 * In this work, we extend this approach to multiple languages and show the effectiveness of cross-lingual pretraining. We
 * propose two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual
 * data, and one supervised that leverages parallel data with a new cross-lingual language model objective. We obtain
 * state-of-the-art results on cross-lingual classification, unsupervised and supervised machine translation. On XNLI, our
 * approach pushes the state of the art by an absolute gain of 4.9% accuracy. On unsupervised machine translation, we
 * obtain 34.3 BLEU on WMT'16 German-English, improving the previous state of the art by more than 9 BLEU. On supervised
 * machine translation, we obtain a new state of the art of 38.5 BLEU on WMT'16 Romanian-English, outperforming the
 * previous best approach by more than 4 BLEU. Our code and pretrained models will be made publicly available.
 *
 * Tips:
 *
 * - XLM has many different checkpoints, which were trained using different objectives: CLM, MLM or TLM. Make sure to
 * select the correct objective for your task (e.g. MLM checkpoints are not suitable for generation).
 * - XLM has multilingual checkpoints which leverage a specific '''lang''' parameter.
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
class XlmEmbeddings(override val uid: String)
  extends AnnotatorModel[XlmEmbeddings]
    with HasBatchedAnnotate[XlmEmbeddings]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("XLM_EMBEDDINGS"))

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  def padTokenId: Int = {
    $$(vocabulary)("<pad>")
  }

  /**
   * Vocabulary used to encode the words to ids with bpeTokenizer.encode
   *
   * @group param
   * */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")


  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /**
   * Holding merges.txt coming from XLM model
   *
   * @group param
   */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges")

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)


  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): XlmEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "XLM models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[TensorflowXlm]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TensorflowWrapper): XlmEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowXlm(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures
          )
        )
      )
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowXlm = _model.get.value

  /** Set Embeddings dimensions for the XLM model
   * Only possible to set this when the first time is saved
   * dimension is not changeable, it comes from XLM config file
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
    dimension -> 1024,
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true
  )

  def tokenizeWithAlignment(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {

    val bpeTokenizer = BpeTokenizer.forModel(
      "xlm",
      merges = $$(merges),
      vocab = $$(vocabulary),
      padWithSentenceTokens = false
    )

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens = tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map { token =>
        val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
        val sentenceBegin = token.begin
        val sentenceEnd = token.end
        val sentenceInedx = tokenIndex.sentenceIndex
        val result = bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceInedx))
        if (result.nonEmpty) result.head else IndexedToken("")
      }
      val wordpieceTokens = bertTokens.flatMap(token => bpeTokenizer.encode(token)).take($(maxSentenceLength))
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
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflowWrapper, "_xlm", XlmEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedXlmModel extends ParamsAndFeaturesReadable[XlmEmbeddings] with HasPretrained[XlmEmbeddings] {
  override val defaultModelName: Some[String] = Some("xlm_clm_ende_1024")

  /** Java compliant-overrides */
  override def pretrained(): XlmEmbeddings = super.pretrained()

  override def pretrained(name: String): XlmEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): XlmEmbeddings = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): XlmEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadXlmTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[XlmEmbeddings] =>

  override val tfFile: String = "xlm_tensorflow"

  def readTensorflow(instance: XlmEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_xlm_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): XlmEmbeddings = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath"
    )

    val vocabFile = new File(tfModelPath + "/assets", "vocab.txt")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(vocabFile.exists(), s"Vocabulary file vocab.txt not found in folder $tfModelPath")

    val mergesFile = new File(tfModelPath + "/assets", "merges.txt")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(mergesFile.exists(), s"merges file merges.txt not found in folder $tfModelPath")

    val vocabResource = new ExternalResource(vocabFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val mergesResource = new ExternalResource(mergesFile.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val merges = ResourceHelper.parseLines(mergesResource)

    val bytePairs: Map[(String, String), Int] = merges.map(_.split(" "))
      .filter(x => x.length > 1 && x.length < 3)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex.toMap

    val (wrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important is we use getSignatures inside setModelIfNotSet */
    new XlmEmbeddings()
      .setVocabulary(words)
      .setMerges(bytePairs)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}


object XlmEmbeddings extends ReadablePretrainedXlmModel with ReadXlmTensorflowModel
