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

import com.johnsnowlabs.ml.ai.XlmRoberta
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ModelArch, ModelEngine, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** Sentence-level embeddings using XLM-RoBERTa. The XLM-RoBERTa model was proposed in
  * [[https://arxiv.org/abs/1911.02116 Unsupervised Cross-lingual Representation Learning at Scale]]
  * by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek,
  * Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based
  * on Facebook's RoBERTa model released in 2019. It is a large multi-lingual language model,
  * trained on 2.5TB of filtered CommonCrawl data.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = XlmRoBertaSentenceEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  * }}}
  * The default model is `"sent_xlm_roberta_base"`, default language is `"xx"` (meaning
  * multi-lingual), if no values are provided. For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/XlmRoBertaSentenceEmbeddingsTestSpec.scala XlmRoBertaSentenceEmbeddingsTestSpec]].
  *
  * '''Paper Abstract:'''
  *
  * ''This paper shows that pretraining multilingual language models at scale leads to significant
  * performance gains for a wide range of cross-lingual transfer tasks. We train a
  * Transformer-based masked language model on one hundred languages, using more than two
  * terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms
  * multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average
  * accuracy on XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R
  * performs particularly well on low-resource languages, improving 11.8% in XNLI accuracy for
  * Swahili and 9.2% for Urdu over the previous XLM model. We also present a detailed empirical
  * evaluation of the key factors that are required to achieve these gains, including the
  * trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high
  * and low resource languages at scale. Finally, we show, for the first time, the possibility of
  * multilingual modeling without sacrificing per-language performance; XLM-Ris very competitive
  * with strong monolingual models on the GLUE and XNLI benchmarks. We will make XLM-R code, data,
  * and models publicly available.''
  *
  * '''Tips:'''
  *   - XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM
  *     multilingual models, it does not require '''lang''' parameter to understand which language
  *     is used, and should be able to determine the correct language from the input ids.
  *   - This implementation is the same as RoBERTa. Refer to the [[RoBertaEmbeddings]] for usage
  *     examples as well as the information relative to the inputs and outputs.
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
  * val sentenceEmbeddings = XlmRoBertaSentenceEmbeddings.pretrained()
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
  * |[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
  * |[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
  * |[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
  * |[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
  * |[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[XlmRoBertaEmbeddings]] for token-level embeddings
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
class XlmRoBertaSentenceEmbeddings(override val uid: String)
    extends AnnotatorModel[XlmRoBertaSentenceEmbeddings]
    with HasBatchedAnnotate[XlmRoBertaSentenceEmbeddings]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("XLM_ROBERTA_EMBEDDINGS"))

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
  def setConfigProtoBytes(bytes: Array[Int]): XlmRoBertaSentenceEmbeddings.this.type =
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
      "XLM-RoBERTa models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[XlmRoberta]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      spp: SentencePieceWrapper): XlmRoBertaSentenceEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new XlmRoberta(
            tensorflowWrapper,
            onnxWrapper,
            None,
            spp,
            $(caseSensitive),
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.sentenceEmbeddings)))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: XlmRoberta = _model.get.value

  /** Set Embeddings dimensions for the XLM-RoBERTa model. Only possible to set this when the
    * first time is saved dimension is not changeable, it comes from XLM-RoBERTa config file.
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
        getModelIfNotSet.predictSequence(sentences, $(batchSize), $(maxSentenceLength))
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
      "_xlmroberta",
      XlmRoBertaSentenceEmbeddings.tfFile,
      configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      "_xlmroberta",
      XlmRoBertaSentenceEmbeddings.sppFile)
  }

}

trait ReadablePretrainedXlmRobertaSentenceModel
    extends ParamsAndFeaturesReadable[XlmRoBertaSentenceEmbeddings]
    with HasPretrained[XlmRoBertaSentenceEmbeddings] {
  override val defaultModelName: Some[String] = Some("sent_xlm_roberta_base")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): XlmRoBertaSentenceEmbeddings = super.pretrained()

  override def pretrained(name: String): XlmRoBertaSentenceEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): XlmRoBertaSentenceEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): XlmRoBertaSentenceEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadXlmRobertaSentenceDLModel extends ReadTensorflowModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[XlmRoBertaSentenceEmbeddings] =>

  override val tfFile: String = "xlmroberta_tensorflow"
  override val sppFile: String = "xlmroberta_spp"

  def readModel(
      instance: XlmRoBertaSentenceEmbeddings,
      path: String,
      spark: SparkSession): Unit = {

    val tfWrapper = readTensorflowModel(path, spark, "_xlmroberta_tf", initAllTables = false)
    val spp = readSentencePieceModel(path, spark, "_xlmroberta_spp", sppFile)
    instance.setModelIfNotSet(spark, Some(tfWrapper), None, spp)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): XlmRoBertaSentenceEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")

    /*Universal parameters for all engines*/
    val annotatorModel = new XlmRoBertaSentenceEmbeddings()

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
          .setModelIfNotSet(spark, Some(tfWrapper), None, spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object XlmRoBertaSentenceEmbeddings
    extends ReadablePretrainedXlmRobertaSentenceModel
    with ReadXlmRobertaSentenceDLModel
