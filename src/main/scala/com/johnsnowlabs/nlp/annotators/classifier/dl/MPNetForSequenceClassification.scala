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

import com.johnsnowlabs.ml.ai.MPNetClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{loadTextAsset, modelSanityCheck, notSupportedEngineError}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** MPNetForSequenceClassification can load MPNet Models with sequence classification/regression
  * head on top (a linear layer on top of the pooled output) e.g. for multi-class document
  * classification tasks.
  *
  * Note that currently, only SetFit models can be imported.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val sequenceClassifier = MPNetForSequenceClassification.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  * }}}
  * The default model is `"mpnet_sequence_classifier_ukr_message"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Text+Classification Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MPNetForSequenceClassificationTestSpec.scala MPNetForSequenceClassificationTestSpec]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  * import spark.implicits._
  *
  * val document = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val sequenceClassifier = MPNetForSequenceClassification
  *   .pretrained()
  *   .setInputCols(Array("document", "token"))
  *   .setOutputCol("label")
  *
  * val texts = Seq(
  *   "I love driving my car.",
  *   "The next bus will arrive in 20 minutes.",
  *   "pineapple on pizza is the worst 🤮")
  * val data = texts.toDF("text")
  *
  * val pipeline = new Pipeline().setStages(Array(document, tokenizer, sequenceClassifier))
  * val pipelineModel = pipeline.fit(data)
  * val results = pipelineModel.transform(data)
  *
  * results.select("label.result").show()
  * +--------------------+
  * |              result|
  * +--------------------+
  * |     [TRANSPORT/CAR]|
  * |[TRANSPORT/MOVEMENT]|
  * |              [FOOD]|
  * +--------------------+
  * }}}
  *
  * @see
  *   [[MPNetForSequenceClassification]] for sequence-level classification
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based classifiers
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
class MPNetForSequenceClassification(override val uid: String)
    extends AnnotatorModel[MPNetForSequenceClassification]
    with HasBatchedAnnotate[MPNetForSequenceClassification]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasCaseSensitiveProperties
    with HasClassifierActivationProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("MPNetForSequenceClassification"))

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

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("<s>")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("</s>")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Labels used to decode predicted IDs back to string tags
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels").setProtected()

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
      "MPNet models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[MPNetClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]
                      ): MPNetForSequenceClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MPNetClassification(
            None,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            tags = $$(labels),
            signatures = getSignatures,
            $$(vocabulary),
            threshold = $(threshold))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: MPNetClassification = _model.get.value

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    set(this.caseSensitive, value)
  }

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> true,
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
    val suffix = "_MPNet_classification"

    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          MPNetForSequenceClassification.onnxFile)

      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          MPNetForSequenceClassification.openvinoFile)
    }

  }

}

trait ReadablePretrainedMPNetForSequenceModel
    extends ParamsAndFeaturesReadable[MPNetForSequenceClassification]
    with HasPretrained[MPNetForSequenceClassification] {
  override val defaultModelName: Some[String] = Some("mpnet_sequence_classifier_ukr_message")

  /** Java compliant-overrides */
  override def pretrained(): MPNetForSequenceClassification = super.pretrained()

  override def pretrained(name: String): MPNetForSequenceClassification =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): MPNetForSequenceClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): MPNetForSequenceClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMPNetForSequenceDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[MPNetForSequenceClassification] =>

  override val onnxFile: String = "mpnet_classification_onnx"
  override val openvinoFile: String = "mpnet_classification_openvino"

  def readModel(
      instance: MPNetForSequenceClassification,
      path: String,
      spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(
            path,
            spark,
            "mpnet_sequence_classification_onnx",
            zipped = true,
            useBundle = false,
            None)
        instance.setModelIfNotSet(spark, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "distilbert_qa_classification_openvino")
        instance.setModelIfNotSet(spark, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MPNetForSequenceClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    val labels = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap

    val annotatorModel = new MPNetForSequenceClassification()
      .setVocabulary(vocabs)
      .setLabels(labels)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        throw new NotImplementedError("Tensorflow Models are currently not supported.")
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrapper), None)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(ovWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MPNetForSequenceClassification]]. Please refer to that class
  * for the documentation.
  */
object MPNetForSequenceClassification
    extends ReadablePretrainedMPNetForSequenceModel
    with ReadMPNetForSequenceDLModel
