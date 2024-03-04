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
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
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
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import com.johnsnowlabs.util.FileHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

import java.nio.file.Files
import java.util.UUID

/** The XLM-RoBERTa model was proposed in
  * [[https://arxiv.org/abs/1911.02116 Unsupervised Cross-lingual Representation Learning at Scale]]
  * by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek,
  * Francisco GuzmÃ¡n, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based
  * on Facebook's RoBERTa model released in 2019. It is a large multi-lingual language model,
  * trained on 2.5TB of filtered CommonCrawl data.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = XlmRoBertaEmbeddings.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"xlm_roberta_base"`, default language is `"xx"` (meaning multi-lingual),
  * if no values are provided. For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/XlmRoBertaEmbeddingsTestSpec.scala XlmRoBertaEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
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
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings
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
  * val embeddings = XlmRoBertaEmbeddings.pretrained()
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
  * |[-0.05969233065843582,-0.030789051204919815,0.04443822056055069,0.09564960747...|
  * |[-0.038839809596538544,0.011712731793522835,0.019954433664679527,0.0667808502...|
  * |[-0.03952755779027939,-0.03455188870429993,0.019103847444057465,0.04311436787...|
  * |[-0.09579929709434509,0.02494969218969345,-0.014753809198737144,0.10259044915...|
  * |[0.004710011184215546,-0.022148698568344116,0.011723337695002556,-0.013356896...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[XlmRoBertaSentenceEmbeddings]] for sentence-level embeddings
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForTokenClassification XlmRoBertaForTokenClassification]]
  *   For XlmRoBerta embeddings with a token classification layer on top
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
class XlmRoBertaEmbeddings(override val uid: String)
    extends AnnotatorModel[XlmRoBertaEmbeddings]
    with HasBatchedAnnotate[XlmRoBertaEmbeddings]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with WriteOnnxModel
    with WriteOpenvinoModel
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
  def setConfigProtoBytes(bytes: Array[Int]): XlmRoBertaEmbeddings.this.type =
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
      openvinoWrapper: Option[OpenvinoWrapper],
      spp: SentencePieceWrapper): XlmRoBertaEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new XlmRoberta(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            spp,
            $(caseSensitive),
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.wordEmbeddings)))
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
    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) =>
        TokenizedWithSentence.unpack(annotations).toArray.map(x => (x, i))
      }

    val sentenceWordEmbeddings =
      getModelIfNotSet.predict(sentencesWithRow.map(_._1), $(batchSize), $(maxSentenceLength))

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
    val suffix = "_xlmroberta"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          XlmRoBertaEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          XlmRoBertaEmbeddings.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          suffix,
          XlmRoBertaEmbeddings.openvinoFile)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      suffix,
      XlmRoBertaEmbeddings.sppFile)
  }

}

trait ReadablePretrainedXlmRobertaModel
    extends ParamsAndFeaturesReadable[XlmRoBertaEmbeddings]
    with HasPretrained[XlmRoBertaEmbeddings] {
  override val defaultModelName: Some[String] = Some("xlm_roberta_base")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): XlmRoBertaEmbeddings = super.pretrained()

  override def pretrained(name: String): XlmRoBertaEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): XlmRoBertaEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): XlmRoBertaEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadXlmRobertaDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel
    with ReadOnnxModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[XlmRoBertaEmbeddings] =>

  override val tfFile: String = "xlmroberta_tensorflow"
  override val onnxFile: String = "xlmroberta_onnx"
  override val openvinoFile: String = "xlmroberta_openvino"
  override val sppFile: String = "xlmroberta_spp"

  def readModel(instance: XlmRoBertaEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_xlmroberta_tf", initAllTables = false)
        val spp = readSentencePieceModel(path, spark, "_xlmroberta_spp", sppFile)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None, spp)

      case ONNX.name => {
        val onnxWrapper =
          readOnnxModel(path, spark, "_xlmroberta_onnx", zipped = true, useBundle = false, None)
        val spp = readSentencePieceModel(path, spark, "_xlmroberta_spp", sppFile)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None, spp)
      }

      case Openvino.name => {
        val openvinoWrapper =
          readOpenvinoModel(path, spark, "_xlmroberta_openvino")
        val spp = readSentencePieceModel(path, spark, "_xlmroberta_spp", sppFile)
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper), spp)
      }
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): XlmRoBertaEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")

    /*Universal parameters for all engines*/
    val annotatorModel = new XlmRoBertaEmbeddings()
    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine

    annotatorModel.set(annotatorModel.engine, modelEngine)

    modelEngine match {
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
          .setModelIfNotSet(spark, Some(tfWrapper), None, None, spModel)

      case ONNX.name =>
        val onnxWrapper = OnnxWrapper.read(localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None, spModel)

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
              zipped = false)
            tmpFolder
          }
        val (ovWrapper: OpenvinoWrapper, tensorNames: Map[String, String]) =
          OpenvinoWrapper.fromOpenvinoFormat(irModelFolder, zipped = false)

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel.setSignatures(tensorNames)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper), spModel)
        FileHelper.delete(tmpFolder)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

object XlmRoBertaEmbeddings extends ReadablePretrainedXlmRobertaModel with ReadXlmRobertaDLModel
