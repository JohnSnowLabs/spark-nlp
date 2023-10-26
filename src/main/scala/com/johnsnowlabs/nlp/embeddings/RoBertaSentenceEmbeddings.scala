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

import com.johnsnowlabs.ml.ai.RoBerta
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
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.BpeTokenizer
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Sentence-level embeddings using RoBERTa. The RoBERTa model was proposed in
  * [[https://arxiv.org/abs/1907.11692 RoBERTa: A Robustly Optimized BERT Pretraining Approach]]
  * by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
  * Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google's BERT model released in
  * 2018.
  *
  * It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining
  * objective and training with much larger mini-batches and learning rates.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = RoBertaSentenceEmbeddings.pretrained()
  *   .setInputCols("sentence")
  *   .setOutputCol("sentence_embeddings")
  * }}}
  * The default model is `"sent_roberta_base"`, if no name is provided. For available pretrained
  * models please see the [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/RoBertaEmbeddingsTestSpec.scala RoBertaEmbeddingsTestSpec]].
  *
  * '''Paper Abstract:'''
  *
  * ''Language model pretraining has led to significant performance gains but careful comparison
  * between different approaches is challenging. Training is computationally expensive, often done
  * on private datasets of different sizes, and, as we will show, hyperparameter choices have
  * significant impact on the final results. We present a replication study of BERT pretraining
  * (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
  * training data size. We find that BERT was significantly undertrained, and can match or exceed
  * the performance of every model published after it. Our best model achieves state-of-the-art
  * results on GLUE, RACE and SQuAD. These results highlight the importance of previously
  * overlooked design choices, and raise questions about the source of recently reported
  * improvements. We release our models and code.''
  *
  * Tips:
  *   - RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same
  *     as GPT-2) and uses a different pretraining scheme.
  *   - RoBERTa doesn't have :obj:`token_type_ids`, you don't need to indicate which token belongs
  *     to which segment. Just separate your segments with the separation token
  *     :obj:`tokenizer.sep_token` (or :obj:`</s>`)
  *
  * The original code can be found ```here```
  * [[https://github.com/pytorch/fairseq/tree/master/examples/roberta]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val sentenceEmbeddings = RoBertaSentenceEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *   .setCaseSensitive(true)
  *
  * // you can either use the output to train ClassifierDL, SentimentDL, or MultiClassifierDL
  * // or you can use EmbeddingsFinisher to prepare the results for Spark ML functions
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     sentenceEmbeddings,
  *     embeddingsFinisher
  *   ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[0.18792399764060974,-0.14591649174690247,0.20547787845134735,0.1468472778797...|
  * |[0.22845706343650818,0.18073144555091858,0.09725798666477203,-0.0417917296290...|
  * |[0.07037967443466187,-0.14801117777824402,-0.03603338822722435,-0.17893412709...|
  * |[-0.08734266459941864,0.2486150562763214,-0.009067727252840996,-0.24408400058...|
  * |[0.22409197688102722,-0.4312366545200348,0.1401449590921402,0.356410235166549...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[RoBertaEmbeddings]] for token-level embeddings
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based embeddings
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
class RoBertaSentenceEmbeddings(override val uid: String)
    extends AnnotatorModel[RoBertaSentenceEmbeddings]
    with HasBatchedAnnotate[RoBertaSentenceEmbeddings]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("RoBertaSentenceEmbeddings"))

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  def padTokenId: Int = {
    $$(vocabulary)("<pad>")
  }

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Holding merges.txt coming from RoBERTa model
    *
    * @group param
    */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

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
  def setConfigProtoBytes(bytes: Array[Int]): RoBertaSentenceEmbeddings.this.type =
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
      "RoBERTa models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[RoBerta]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): RoBertaSentenceEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new RoBerta(
            tensorflowWrapper,
            onnxWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.sentenceEmbeddings)))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: RoBerta = _model.get.value

  /** Set Embeddings dimensions for the RoBERTa model. Only possible to set this when the first
    * time is saved dimension is not changeable, it comes from RoBERTa config file.
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

  setDefault(dimension -> 768, batchSize -> 8, maxSentenceLength -> 128, caseSensitive -> true)

  def tokenize(sentences: Seq[Sentence]): Seq[WordpieceTokenizedSentence] = {

    val bpeTokenizer = BpeTokenizer.forModel(
      "roberta",
      merges = $$(merges),
      vocab = $$(vocabulary))

    sentences.map { s =>
      // filter empty and only whitespace tokens
      val content = if ($(caseSensitive)) s.content else s.content.toLowerCase()
      val sentenceBegin = s.start
      val sentenceEnd = s.end
      val sentenceIndex = s.index
      val tokens =
        bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
      val wordpieceTokens =
        tokens.flatMap(token => bpeTokenizer.encode(token)).take($(maxSentenceLength))
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

    /*Return empty if the real sentences are empty*/
    batchedAnnotations.map(annotations => {
      val sentences = SentenceSplit.unpack(annotations).toArray

      if (sentences.nonEmpty) {
        val tokenized = tokenize(sentences)
        getModelIfNotSet.predictSequence(tokenized, sentences, $(batchSize), $(maxSentenceLength))
      } else {
        Seq.empty[Annotation]
      }
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

  /** Input Annotator Types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

  /** Output Annotator Types: WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper.get,
      "_roberta",
      RoBertaSentenceEmbeddings.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedRobertaSentenceModel
    extends ParamsAndFeaturesReadable[RoBertaSentenceEmbeddings]
    with HasPretrained[RoBertaSentenceEmbeddings] {
  override val defaultModelName: Some[String] = Some("sent_roberta_base")

  /** Java compliant-overrides */
  override def pretrained(): RoBertaSentenceEmbeddings = super.pretrained()

  override def pretrained(name: String): RoBertaSentenceEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): RoBertaSentenceEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): RoBertaSentenceEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadRobertaSentenceDLModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[RoBertaSentenceEmbeddings] =>

  override val tfFile: String = "roberta_tensorflow"

  def readModel(instance: RoBertaSentenceEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_roberta_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, Some(tf), None)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): RoBertaSentenceEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new RoBertaSentenceEmbeddings()
      .setVocabulary(vocabs)
      .setMerges(bytePairs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
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
          .setModelIfNotSet(spark, Some(wrapper), None)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[RoBertaSentenceEmbeddings]]. Please refer to that class for
  * the documentation.
  */
object RoBertaSentenceEmbeddings
    extends ReadablePretrainedRobertaSentenceModel
    with ReadRobertaSentenceDLModel
