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
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{FloatParam, IntArrayParam, Param, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}

/** SentimentDL, an annotator for multi-class sentiment analysis.
  *
  * In natural language processing, sentiment analysis is the task of classifying the affective
  * state or subjective view of a text. A common example is if either a product review or tweet
  * can be interpreted positively or negatively.
  *
  * This is the instantiated model of the [[SentimentDLApproach]]. For training your own model,
  * please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val sentiment = SentimentDLModel.pretrained()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCol("sentiment")
  * }}}
  * The default model is `"sentimentdl_use_imdb"`, if no name is provided. It is english sentiment
  * analysis trained on the IMDB dataset. For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Sentiment+Analysis Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLTestSpec.scala SentimentDLTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
  * import com.johnsnowlabs.nlp.annotators.classifier.dl.SentimentDLModel
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val useEmbeddings = UniversalSentenceEncoder.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *
  * val sentiment = SentimentDLModel.pretrained("sentimentdl_use_twitter")
  *   .setInputCols("sentence_embeddings")
  *   .setThreshold(0.7F)
  *   .setOutputCol("sentiment")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   useEmbeddings,
  *   sentiment
  * ))
  *
  * val data = Seq(
  *   "Wow, the new video is awesome!",
  *   "bruh what a damn waste of time"
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("text", "sentiment.result").show(false)
  * +------------------------------+----------+
  * |text                          |result    |
  * +------------------------------+----------+
  * |Wow, the new video is awesome!|[positive]|
  * |bruh what a damn waste of time|[negative]|
  * +------------------------------+----------+
  * }}}
  *
  * @see
  *   [[ClassifierDLModel]] for general single-class classification
  * @see
  *   [[MultiClassifierDLModel]] for general multi-class classification
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
class SentimentDLModel(override val uid: String)
    extends AnnotatorModel[SentimentDLModel]
    with HasSimpleAnnotate[SentimentDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable {
  def this() = this(Identifiable.randomUID("SentimentDLModel"))

  /** Input Annotator Types: SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  /** Output Annotator Types: CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = CATEGORY

  /** The minimum threshold for the final result otherwise it will be either neutral or the value
    * set in thresholdLabel (Default: `0.6f`)
    *
    * @group param
    */
  val threshold = new FloatParam(
    this,
    "threshold",
    "The minimum threshold for the final result otherwise it will be either neutral or the value set in thresholdLabel.s")

  /** In case the score is less than threshold, what should be the label (Default: `"neutral"`)
    *
    * @group param
    */
  val thresholdLabel = new Param[String](
    this,
    "thresholdLabel",
    "In case the score is less than threshold, what should be the label. Default is neutral.")

  /** @group setParam */
  def setThreshold(threshold: Float): SentimentDLModel.this.type = set(this.threshold, threshold)

  /** @group setParam */
  def setThresholdLabel(label: String): SentimentDLModel.this.type =
    set(this.thresholdLabel, label)

  /** @group getParam */
  def getThreshold: Float = $(this.threshold)

  /** @group getParam */
  def getThresholdLabel: String = $(this.thresholdLabel)

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
  def setConfigProtoBytes(bytes: Array[Int]): SentimentDLModel.this.type =
    set(this.configProtoBytes, bytes)

  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Dataset Params
    *
    * @group param
    */
  val datasetParams = new StructFeature[ClassifierDatasetEncoderParams](this, "datasetParams")

  /** @group setParam */
  def setDatasetParams(params: ClassifierDatasetEncoderParams): SentimentDLModel.this.type =
    set(this.datasetParams, params)

  /** Labels that the model was trained with
    *
    * @group param
    */
  val classes =
    new StringArrayParam(this, "classes", "keep an internal copy of classes for Python")

  private var _model: Option[Broadcast[TensorflowSentiment]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tf: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)

      _model = Some(
        spark.sparkContext.broadcast(new TensorflowSentiment(tf, encoder, Verbose.Silent)))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowSentiment = _model.get.value

  /** get the tags used to trained this NerDLModel
    *
    * @group getParam
    */
  def getClasses: Array[String] = {
    val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)
    set(classes, encoder.tags)
    encoder.tags
  }

  setDefault(threshold -> 0.6f, thresholdLabel -> "neutral")

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateStorageRef(dataset, $(inputCols), AnnotatorType.SENTENCE_EMBEDDINGS)
    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = annotations
      .filter(_.annotatorType == SENTENCE_EMBEDDINGS)
      .groupBy(_.metadata.getOrElse[String]("sentence", "0").toInt)
      .toSeq
      .sortBy(_._1)

    if (sentences.nonEmpty)
      getModelIfNotSet.predict(sentences, getConfigProtoBytes, $(threshold), $(thresholdLabel))
    else Seq.empty[Annotation]

  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_sentimentdl",
      SentimentDLModel.tfFile,
      configProtoBytes = getConfigProtoBytes)

  }
}

trait ReadablePretrainedSentimentDL
    extends ParamsAndFeaturesReadable[SentimentDLModel]
    with HasPretrained[SentimentDLModel] {
  override val defaultModelName: Some[String] = Some("sentimentdl_use_imdb")

  override def pretrained(name: String, lang: String, remoteLoc: String): SentimentDLModel = {
    ResourceDownloader.downloadModel(SentimentDLModel, name, Option(lang), remoteLoc)
  }

  /** Java compliant-overrides */
  override def pretrained(): SentimentDLModel =
    pretrained(defaultModelName.get, defaultLang, defaultLoc)

  override def pretrained(name: String): SentimentDLModel =
    pretrained(name, defaultLang, defaultLoc)

  override def pretrained(name: String, lang: String): SentimentDLModel =
    pretrained(name, lang, defaultLoc)
}

trait ReadSentimentDLTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[SentimentDLModel] =>

  override val tfFile: String = "sentimentdl_tensorflow"

  def readTensorflow(instance: SentimentDLModel, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_sentimentdl_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
    // This allows for Python to access getClasses function
    val encoder = new ClassifierDatasetEncoder(instance.datasetParams.get.get)
    instance.set(instance.classes, encoder.tags)
  }

  addReader(readTensorflow)
}

/** This is the companion object of [[SentimentDLModel]]. Please refer to that class for the
  * documentation.
  */
object SentimentDLModel extends ReadablePretrainedSentimentDL with ReadSentimentDLTensorflowModel
