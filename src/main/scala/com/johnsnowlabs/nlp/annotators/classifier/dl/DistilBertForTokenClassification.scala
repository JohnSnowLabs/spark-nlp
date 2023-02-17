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
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ModelEngine
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** DistilBertForTokenClassification can load Bert Models with a token classification head on top
  * (a linear layer on top of the hidden-states output) e.g. for Named-Entity-Recognition (NER)
  * tasks.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val tokenClassifier = DistilBertForTokenClassification.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  * }}}
  * The default model is `"distilbert_base_token_classifier_conll03"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]] and to see more extended
  * examples, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/DistilBertForTokenClassificationTestSpec.scala DistilBertForTokenClassificationTestSpec]].
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
  * val tokenClassifier = DistilBertForTokenClassification.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  *   .setCaseSensitive(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   tokenClassifier
  * ))
  *
  * val data = Seq("John Lenon was born in London and lived in Paris. My name is Sarah and I live in London").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("label.result").show(false)
  * +------------------------------------------------------------------------------------+
  * |result                                                                              |
  * +------------------------------------------------------------------------------------+
  * |[B-PER, I-PER, O, O, O, B-LOC, O, O, O, B-LOC, O, O, O, O, B-PER, O, O, O, O, B-LOC]|
  * +------------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[DistilBertForTokenClassification]] for token-level classification
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
class DistilBertForTokenClassification(override val uid: String)
    extends AnnotatorModel[DistilBertForTokenClassification]
    with HasBatchedAnnotate[DistilBertForTokenClassification]
    with WriteTensorflowModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("DISTILBERT_FOR_TOKEN_CLASSIFICATION"))

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
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.NAMED_ENTITY

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

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
  def setConfigProtoBytes(bytes: Array[Int]): DistilBertForTokenClassification.this.type =
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
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[TensorflowDistilBertClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper): DistilBertForTokenClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowDistilBertClassification(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            signatures = getSignatures,
            $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowDistilBertClassification = _model.get.value

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(batchSize -> 8, maxSentenceLength -> 128, caseSensitive -> true)

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
    val batchedTokenizedSentences: Array[Array[TokenizedSentence]] = batchedAnnotations
      .map(annotations => TokenizedWithSentence.unpack(annotations).toArray)
      .toArray
    /*Return empty if the real tokens are empty*/
    if (batchedTokenizedSentences.nonEmpty) batchedTokenizedSentences.map(tokenizedSentences => {

      getModelIfNotSet.predict(
        tokenizedSentences,
        $(batchSize),
        $(maxSentenceLength),
        $(caseSensitive),
        $$(labels))
    })
    else {
      Seq(Seq.empty[Annotation])
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper,
      "_distilbert_classification",
      DistilBertForTokenClassification.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedDistilBertForTokenModel
    extends ParamsAndFeaturesReadable[DistilBertForTokenClassification]
    with HasPretrained[DistilBertForTokenClassification] {
  override val defaultModelName: Some[String] = Some("distilbert_base_token_classifier_conll03")

  /** Java compliant-overrides */
  override def pretrained(): DistilBertForTokenClassification = super.pretrained()

  override def pretrained(name: String): DistilBertForTokenClassification = super.pretrained(name)

  override def pretrained(name: String, lang: String): DistilBertForTokenClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): DistilBertForTokenClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadDistilBertForTokenDLModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[DistilBertForTokenClassification] =>

  override val tfFile: String = "distilbert_classification_tensorflow"

  def readModel(
      instance: DistilBertForTokenClassification,
      path: String,
      spark: SparkSession): Unit = {

    val tf =
      readTensorflowModel(path, spark, "_distilbert_classification_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): DistilBertForTokenClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    val labels = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap

    val annotatorModel = new DistilBertForTokenClassification()
      .setVocabulary(vocabs)
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
          .setModelIfNotSet(spark, wrapper)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[DistilBertForTokenClassification]]. Please refer to that
  * class for the documentation.
  */
object DistilBertForTokenClassification
    extends ReadablePretrainedDistilBertForTokenModel
    with ReadDistilBertForTokenDLModel
