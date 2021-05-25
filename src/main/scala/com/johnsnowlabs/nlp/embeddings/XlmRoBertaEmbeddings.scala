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
import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.io.File

/**
 * The XLM-RoBERTa model was proposed in '''Unsupervised Cross-lingual Representation Learning at Scale'''
 * [[https://arxiv.org/abs/1911.02116]] by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
 * Wenzek, Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
 * RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
 * data.
 *
 * The abstract from the paper is the following:
 *
 * This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a
 * wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
 * languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
 * outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on
 * XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
 * low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We
 * also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the
 * trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource
 * languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing
 * per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We
 * will make XLM-R code, data, and models publicly available.
 *
 * Tips:
 *
 * - XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
 * not require '''lang''' parameter to understand which language is used, and should be able to determine the correct
 * language from the input ids.
 * - This implementation is the same as RoBERTa. Refer to the [[com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings]] for usage examples
 * as well as the information relative to the inputs and outputs.
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
class XlmRoBertaEmbeddings(override val uid: String)
  extends AnnotatorModel[XlmRoBertaEmbeddings]
    with HasBatchedAnnotate[XlmRoBertaEmbeddings]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("XLM_ROBERTA_EMBEDDINGS"))

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    0
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    2
  }

  def padTokenId: Int = {
    1
  }

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): XlmRoBertaEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "XLM-RoBERTa models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[TensorflowXlmRoberta]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflowWrapper: TensorflowWrapper, spp: SentencePieceWrapper): XlmRoBertaEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowXlmRoberta(
            tensorflowWrapper,
            spp,
            $(batchSize),
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
  def getModelIfNotSet: TensorflowXlmRoberta = _model.get.value

  /** Set Embeddings dimensions for the XLM-RoBERTa model
   * Only possible to set this when the first time is saved
   * dimension is not changeable, it comes from XLM-RoBERTa config file
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
    caseSensitive -> true
  )

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

      val embeddings = getModelIfNotSet.calculateEmbeddings(
        tokenizedSentences,
        $(batchSize),
        $(maxSentenceLength),
        $(caseSensitive)
      )
      WordpieceEmbeddingsSentence.pack(embeddings)
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
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflowWrapper, "_xlmroberta", XlmRoBertaEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_xlmroberta", XlmRoBertaEmbeddings.sppFile)
  }

}

trait ReadablePretrainedXlmRobertaModel extends ParamsAndFeaturesReadable[XlmRoBertaEmbeddings] with HasPretrained[XlmRoBertaEmbeddings] {
  override val defaultModelName: Some[String] = Some("xlm_roberta_base")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): XlmRoBertaEmbeddings = super.pretrained()

  override def pretrained(name: String): XlmRoBertaEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): XlmRoBertaEmbeddings = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): XlmRoBertaEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadXlmRobertaTensorflowModel extends ReadTensorflowModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[XlmRoBertaEmbeddings] =>

  override val tfFile: String = "xlmroberta_tensorflow"
  override val sppFile: String = "xlmroberta_spp"

  def readTensorflow(instance: XlmRoBertaEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_xlmroberta_tf", initAllTables = false)
    val spp = readSentencePieceModel(path, spark, "_xlmroberta_spp", sppFile)
    instance.setModelIfNotSet(spark, tf, spp)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): XlmRoBertaEmbeddings = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")
    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath"
    )
    val sppModelPath = tfModelPath + "/assets"
    val sppModel = new File(sppModelPath, "sentencepiece.bpe.model")
    require(sppModel.exists(), s"SentencePiece model 30k-clean.model not found in folder $sppModelPath")


    val (wrapper, signatures) = TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)
    val spp = SentencePieceWrapper.read(sppModel.toString)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important is we use getSignatures inside setModelIfNotSet */
    new XlmRoBertaEmbeddings()
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper, spp)
  }
}


object XlmRoBertaEmbeddings extends ReadablePretrainedXlmRobertaModel with ReadXlmRobertaTensorflowModel
