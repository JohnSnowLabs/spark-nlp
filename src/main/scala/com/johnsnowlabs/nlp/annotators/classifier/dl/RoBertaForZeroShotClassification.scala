/*
 * Copyright 2017-2023 John Snow Labs
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

import com.johnsnowlabs.ml.ai.RoBertaClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

/** RoBertaForZeroShotClassification using a `ModelForSequenceClassification` trained on NLI
  * (natural language inference) tasks. Equivalent of `RoBertaForZeroShotClassification ` models,
  * but these models don't require a hardcoded number of potential classes, they can be chosen at
  * runtime. It usually means it's slower but it is much more flexible.
  *
  * Note that the model will loop through all provided labels. So the more labels you have, the
  * longer this process will take.
  *
  * Any combination of sequences and labels can be passed and each combination will be posed as a
  * premise/hypothesis pair and passed to the pretrained model.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val sequenceClassifier = RoBertaForZeroShotClassification .pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("label")
  * }}}
  * The default model is `"roberta_base_zero_shot_classifier_nli"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Text+Classification Models Hub]].
  *
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
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
  * val sequenceClassifier = RoBertaForZeroShotClassification .pretrained()
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
  *   [[RoBertaForZeroShotClassification]] for sequence-level classification
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
class RoBertaForZeroShotClassification(override val uid: String)
    extends AnnotatorModel[RoBertaForZeroShotClassification]
    with HasBatchedAnnotate[RoBertaForZeroShotClassification]
    with WriteTensorflowModel
    with WriteOnnxModel
    with HasCaseSensitiveProperties
    with HasClassifierActivationProperties
    with HasEngine
    with HasCandidateLabelsProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("ROBERTA_FOR_ZERO_SHOT_CLASSIFICATION"))

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

  def padTokenId: Int = {
    $$(vocabulary)("<pad>")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = {
    set(vocabulary, value)
    this
  }

  /** Labels used to decode predicted IDs back to string tags
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels")

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = {
    if (get(labels).isEmpty)
      set(labels, value)
    this
  }

  /** Returns labels used to train this model */
  def getClasses: Array[String] = {
    $$(labels).keys.toArray
  }

  /** Holding merges.txt coming from RoBERTa model
    *
    * @group param
    */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges")

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document
    * by averaging probabilities in all sentences (Default: `false`).
    *
    * Due to max sequence length limit in almost all transformer models such as RoBerta (512
    * tokens), this parameter helps feeding all the sentences into the model and averaging all the
    * probabilities for the entire document instead of probabilities per sentence.
    *
    * @group param
    */
  val coalesceSentences = new BooleanParam(
    this,
    "coalesceSentences",
    "If sets to true the output of  all sentences will be averaged to one output instead of one output per sentence. Defaults to false.")

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
  def setConfigProtoBytes(bytes: Array[Int]): RoBertaForZeroShotClassification.this.type =
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
      "RoBerta models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  private var _model: Option[Broadcast[RoBertaClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): RoBertaForZeroShotClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new RoBertaClassification(
            tensorflowWrapper,
            onnxWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            padTokenId,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            signatures = getSignatures,
            $$(merges),
            $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: RoBertaClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `true`).
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
        getModelIfNotSet.predictSequenceWithZeroShot(
          tokenizedSentences,
          sentences,
          $(candidateLabels),
          $(entailmentIdParam),
          $(contradictionIdParam),
          $(batchSize),
          $(maxSentenceLength),
          $(caseSensitive),
          $(coalesceSentences),
          $$(labels),
          getActivation)

      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          "_roberta_classification",
          RoBertaForZeroShotClassification.tfFile,
          configProtoBytes = getConfigProtoBytes)

      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          "_roberta_classification",
          RoBertaForZeroShotClassification.onnxFile)
    }
  }
}

trait ReadablePretrainedRoBertaForZeroShotModel
    extends ParamsAndFeaturesReadable[RoBertaForZeroShotClassification]
    with HasPretrained[RoBertaForZeroShotClassification] {
  override val defaultModelName: Some[String] = Some("roberta_base_zero_shot_classifier_nli")

  /** Java compliant-overrides */
  override def pretrained(): RoBertaForZeroShotClassification = super.pretrained()

  override def pretrained(name: String): RoBertaForZeroShotClassification =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): RoBertaForZeroShotClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): RoBertaForZeroShotClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadRoBertaForZeroShotDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[RoBertaForZeroShotClassification] =>

  override val tfFile: String = "roberta_classification_tensorflow"
  override val onnxFile: String = "roberta_classification_onnx"

  def readModel(
      instance: RoBertaForZeroShotClassification,
      path: String,
      spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper =
          readTensorflowModel(path, spark, "_roberta_classification_tf")
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(
            path,
            spark,
            "_deberta_classification_onnx",
            zipped = true,
            useBundle = false,
            None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)

    }
  }
  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): RoBertaForZeroShotClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
    val labels = loadTextAsset(localModelPath, "labels.txt").zipWithIndex.toMap
    val bytePairs = loadTextAsset(localModelPath, "merges.txt")
      .map(_.split(" "))
      .filter(w => w.length == 2)
      .map { case Array(c1, c2) => (c1, c2) }
      .zipWithIndex
      .toMap

    val entailmentIds = labels.filter(x => x._1.toLowerCase().startsWith("entail")).values.toArray
    val contradictionIds =
      labels.filter(x => x._1.toLowerCase().startsWith("contradict")).values.toArray

    require(
      entailmentIds.length == 1 && contradictionIds.length == 1,
      s"""This annotator supports classifiers trained on NLI datasets. You must have only at least 2 or maximum 3 labels in your dataset:

          example with 3 labels: 'contradict', 'neutral', 'entailment'
          example with 2 labels: 'contradict', 'entailment'

          You can modify assets/labels.txt file to match the above format.

          Current labels: ${labels.keys.mkString(", ")}
          """)

    val annotatorModel = new RoBertaForZeroShotClassification()
      .setVocabulary(vocabs)
      .setLabels(labels)
      .setMerges(bytePairs)
      .setCandidateLabels(labels.keys.toArray)

    /* set the entailment id */
    annotatorModel.set(annotatorModel.entailmentIdParam, entailmentIds.head)
    /* set the contradiction id */
    annotatorModel.set(annotatorModel.contradictionIdParam, contradictionIds.head)
    /* set the engine */
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
          .setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[RoBertaForZeroShotClassification]]. Please refer to that
  * class for the documentation.
  */
object RoBertaForZeroShotClassification
    extends ReadablePretrainedRoBertaForZeroShotModel
    with ReadRoBertaForZeroShotDLModel
