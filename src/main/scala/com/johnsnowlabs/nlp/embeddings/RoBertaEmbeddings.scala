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

/** The RoBERTa model was proposed in
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
  * val embeddings = RoBertaEmbeddings.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"roberta_base"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/RoBertaEmbeddingsTestSpec.scala RoBertaEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
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
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings
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
  * val embeddings = RoBertaEmbeddings.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("embeddings")
  *   .setCaseSensitive(true)
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     embeddings,
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
  *   [[RoBertaSentenceEmbeddings]] for sentence-level embeddings
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForTokenClassification RoBertaForTokenClassification]]
  *   For RoBerta embeddings with a token classification layer on top
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
class RoBertaEmbeddings(override val uid: String)
    extends AnnotatorModel[RoBertaEmbeddings]
    with HasBatchedAnnotate[RoBertaEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("ROBERTA_EMBEDDINGS"))

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
  def setConfigProtoBytes(bytes: Array[Int]): RoBertaEmbeddings.this.type =
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
      onnxWrapper: Option[OnnxWrapper]): RoBertaEmbeddings = {
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
            modelArch = ModelArch.wordEmbeddings)))
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

  def tokenizeWithAlignment(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {

    val bpeTokenizer =
      BpeTokenizer.forModel("roberta", merges = $$(merges), vocab = $$(vocabulary))

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result =
              bpeTokenizer.tokenize(Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordpieceTokens =
        bertTokens.flatMap(token => bpeTokenizer.encode(token)).take($(maxSentenceLength))
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

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_roberta"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          RoBertaEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          RoBertaEmbeddings.onnxFile)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }
}

trait ReadablePretrainedRobertaModel
    extends ParamsAndFeaturesReadable[RoBertaEmbeddings]
    with HasPretrained[RoBertaEmbeddings] {
  override val defaultModelName: Some[String] = Some("roberta_base")

  /** Java compliant-overrides */
  override def pretrained(): RoBertaEmbeddings = super.pretrained()

  override def pretrained(name: String): RoBertaEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): RoBertaEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): RoBertaEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadRobertaDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[RoBertaEmbeddings] =>

  override val tfFile: String = "roberta_tensorflow"
  override val onnxFile: String = "roberta_onnx"

  def readModel(instance: RoBertaEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_roberta_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name => {
        val onnxWrapper =
          readOnnxModel(path, spark, "_roberta_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
      }
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): RoBertaEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new RoBertaEmbeddings()
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

/** This is the companion object of [[RoBertaEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object RoBertaEmbeddings extends ReadablePretrainedRobertaModel with ReadRobertaDLModel
