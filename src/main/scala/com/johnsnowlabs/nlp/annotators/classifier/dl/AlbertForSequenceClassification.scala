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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ModelEngine
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** AlbertForSequenceClassification can load ALBERT Models with sequence classification/regression
  * head on top (a linear layer on top of the pooled output) e.g. for multi-class document
  * classification tasks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val sequenceClassifier = AlbertForSequenceClassification.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  * }}}
  * The default model is `"albert_base_sequence_classifier_imdb"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Text+Classification Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/AlbertForSequenceClassificationTestSpec.scala AlbertForSequenceClassification]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
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
  * val sequenceClassifier = AlbertForSequenceClassification.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  *   .setCaseSensitive(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   sequenceClassifier
  * ))
  *
  * val data = Seq("I loved this movie when I was a child.", "It was pretty boring.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("label.result").show(false)
  * +------+
  * |result|
  * +------+
  * |[pos] |
  * |[neg] |
  * +------+
  * }}}
  *
  * @see
  *   [[AlbertForSequenceClassification]] for sequence-level classification
  * @see
  *   [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of
  *   transformer based classifiers
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
class AlbertForSequenceClassification(override val uid: String)
    extends AnnotatorModel[AlbertForSequenceClassification]
    with HasBatchedAnnotate[AlbertForSequenceClassification]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with HasCaseSensitiveProperties
    with HasClassifierActivationProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("AlbertForSequenceClassification"))

  /** Input Annotator Types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Output Annotator Types: CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  /** Labels used to decode predicted IDs back to string tags
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels")

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** Returns labels used to train this model */
  def getClasses: Array[String] = {
    $$(labels).keys.toArray
  }

  /** Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document
    * by averaging probabilities in all sentences (Default: `false`).
    *
    * Due to max sequence length limit in almost all transformer models such as BERT (512 tokens),
    * this parameter helps feeding all the sentences into the model and averaging all the
    * probabilities for the entire document instead of probabilities per sentence.
    *
    * @group param
    */
  val coalesceSentences = new BooleanParam(
    this,
    "coalesceSentences",
    "If sets to true the output of all sentences will be averaged to one output instead of one output per sentence. Default to true.")

  /** @group setParam */
  def setCoalesceSentences(value: Boolean): this.type = set(coalesceSentences, value)

  /** @group getParam */
  def getCoalesceSentences: Boolean = $(coalesceSentences)

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
  def setConfigProtoBytes(bytes: Array[Int]): AlbertForSequenceClassification.this.type =
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
      "ALBERT models do not support sequences longer than 512 because of trainable positional embeddings.")
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
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowAlbertClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper,
      spp: SentencePieceWrapper): AlbertForSequenceClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowAlbertClassification(
            tensorflowWrapper,
            spp,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            signatures = getSignatures)))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowAlbertClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `false`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> false,
    coalesceSentences -> false)

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
    batchedAnnotations.map(annotations => {
      val sentences = SentenceSplit.unpack(annotations).toArray
      val tokenizedSentences = TokenizedWithSentence.unpack(annotations).toArray

      if (tokenizedSentences.nonEmpty) {
        getModelIfNotSet.predictSequence(
          tokenizedSentences,
          sentences,
          $(batchSize),
          $(maxSentenceLength),
          $(caseSensitive),
          $(coalesceSentences),
          $$(labels),
          $(activation))
      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper,
      "_albert_classification",
      AlbertForSequenceClassification.tfFile,
      configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(
      path,
      spark,
      getModelIfNotSet.spp,
      "_albert",
      AlbertForSequenceClassification.sppFile)
  }
}

trait ReadablePretrainedAlbertForSequenceModel
    extends ParamsAndFeaturesReadable[AlbertForSequenceClassification]
    with HasPretrained[AlbertForSequenceClassification] {
  override val defaultModelName: Some[String] = Some("albert_base_sequence_classifier_imdb")

  /** Java compliant-overrides */
  override def pretrained(): AlbertForSequenceClassification = super.pretrained()

  override def pretrained(name: String): AlbertForSequenceClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): AlbertForSequenceClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): AlbertForSequenceClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAlbertForSequenceDLModel
    extends ReadTensorflowModel
    with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[AlbertForSequenceClassification] =>

  override val tfFile: String = "albert_classification_tensorflow"
  override val sppFile: String = "albert_spp"

  def readModel(
      instance: AlbertForSequenceClassification,
      path: String,
      spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_albert_classification_tf", initAllTables = false)
    val spp = readSentencePieceModel(path, spark, "_albert_spp", sppFile)
    instance.setModelIfNotSet(spark, tf, spp)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AlbertForSequenceClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "spiece.model")
    val labels = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap

    val annotatorModel = new AlbertForSequenceClassification()
      .setLabels(labels)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ModelEngine.tensorflow =>
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
          .setModelIfNotSet(spark, wrapper, spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[AlbertForSequenceClassification]]. Please refer to that
  * class for the documentation.
  */
object AlbertForSequenceClassification
    extends ReadablePretrainedAlbertForSequenceModel
    with ReadAlbertForSequenceDLModel
