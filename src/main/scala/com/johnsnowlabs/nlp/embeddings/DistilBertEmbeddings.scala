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

import com.johnsnowlabs.ml.ai.DistilBert
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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT
  * base. It has 40% less parameters than `bert-base-uncased`, runs 60% faster while preserving
  * over 95% of BERT's performances as measured on the GLUE language understanding benchmark.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = DistilBertEmbeddings.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"distilbert_base_cased"`, if no name is provided. For available
  * pretrained models please see the [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/DistilBertEmbeddingsTestSpec.scala DistilBertEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * The DistilBERT model was proposed in the paper
  * [[https://arxiv.org/abs/1910.01108 DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter]].
  *
  * '''Paper Abstract:'''
  *
  * ''As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural
  * Language Processing (NLP), operating these large models in on-the-edge and/or under
  * constrained computational training or inference budgets remains challenging. In this work, we
  * propose a method to pre-train a smaller general-purpose language representation model, called
  * DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like
  * its larger counterparts. While most prior work investigated the use of distillation for
  * building task-specific models, we leverage knowledge distillation during the pretraining phase
  * and show that it is possible to reduce the size of a BERT model by 40%, while retaining 97% of
  * its language understanding capabilities and being 60% faster. To leverage the inductive biases
  * learned by larger models during pretraining, we introduce a triple loss combining language
  * modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is
  * cheaper to pre-train and we demonstrate its capabilities for on-device computations in a
  * proof-of-concept experiment and a comparative on-device study.''
  *
  * Tips:
  *   - DistilBERT doesn't have `:obj:token_type_ids`, you don't need to indicate which token
  *     belongs to which segment. Just separate your segments with the separation token
  *     `:obj:tokenizer.sep_token` (or `:obj:[SEP]`).
  *   - DistilBERT doesn't have options to select the input positions (`:obj:position_ids` input).
  *     This could be added if necessary though, just let us know if you need this option.
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings
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
  * val embeddings = DistilBertEmbeddings.pretrained()
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
  * |[0.1127224713563919,-0.1982710212469101,0.5360898375511169,-0.272536993026733...|
  * |[0.35534414649009705,0.13215228915214539,0.40981462597846985,0.14036104083061...|
  * |[0.328085333108902,-0.06269335001707077,-0.017595693469047546,-0.024373905733...|
  * |[0.15617232024669647,0.2967822253704071,0.22324979305267334,-0.04568954557180...|
  * |[0.45411425828933716,0.01173491682857275,0.190129816532135,0.1178255230188369...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForTokenClassification DistilBertForTokenClassification]]
  *   for DistilBertEmbeddings with a token classification layer on top
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForSequenceClassification DistilBertForSequenceClassification]]
  *   for DistilBertEmbeddings with a sequence classification layer on top
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
class DistilBertEmbeddings(override val uid: String)
    extends AnnotatorModel[DistilBertEmbeddings]
    with HasBatchedAnnotate[DistilBertEmbeddings]
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
  def this() = this(Identifiable.randomUID("DISTILBERT_EMBEDDINGS"))

  /** Input Annotator Types: DOCUMENT. TOKEN
    *
    * @group param
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Output Annotator Types: WORD_EMBEDDINGS
    *
    * @group param
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

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
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): DistilBertEmbeddings.this.type =
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
      "DistilBERT models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[DistilBert]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): DistilBertEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new DistilBert(
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

  /** @group getParam */
  def getModelIfNotSet: DistilBert = _model.get.value

  /** Set Embeddings dimensions for the DistilBERT model. Only possible to set this when the first
    * time is saved dimension is not changeable, it comes from DistilBERT config file.
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
    val suffix = "_distilbert"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          DistilBertEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          DistilBertEmbeddings.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          DistilBertEmbeddings.openvinoFile)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

}

trait ReadablePretrainedDistilBertModel
    extends ParamsAndFeaturesReadable[DistilBertEmbeddings]
    with HasPretrained[DistilBertEmbeddings] {
  override val defaultModelName: Some[String] = Some("distilbert_base_cased")

  /** Java compliant-overrides */
  override def pretrained(): DistilBertEmbeddings = super.pretrained()

  override def pretrained(name: String): DistilBertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): DistilBertEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): DistilBertEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadDistilBertDLModel
    extends ReadTensorflowModel
    with ReadOnnxModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[DistilBertEmbeddings] =>

  override val tfFile: String = "distilbert_tensorflow"
  override val onnxFile: String = "distilbert_onnx"
  override val openvinoFile: String = "distilbert_openvino"

  def readModel(instance: DistilBertEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_distilbert_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_distilbert_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_distilbert_openvino")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): DistilBertEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new DistilBertEmbeddings()
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
          .setModelIfNotSet(spark, Some(tfWrapper), None, None)

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

/** This is the companion object of [[DistilBertEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object DistilBertEmbeddings extends ReadablePretrainedDistilBertModel with ReadDistilBertDLModel
