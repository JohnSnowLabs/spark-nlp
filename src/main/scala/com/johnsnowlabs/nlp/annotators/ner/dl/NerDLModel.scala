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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}

/** This Named Entity recognition annotator is a generic NER model based on Neural Networks.
  *
  * Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most
  * datasets.
  *
  * This is the instantiated model of the [[NerDLApproach]]. For training your own model, please
  * see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val nerModel = NerDLModel.pretrained()
  *   .setInputCols("sentence", "token", "embeddings")
  *   .setOutputCol("ner")
  * }}}
  * The default model is `"ner_dl"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition Models Hub]].
  * Additionally, pretrained pipelines are available for this module, see
  * [[https://nlp.johnsnowlabs.com/docs/en/pipelines Pipelines]].
  *
  * Note that some pretrained models require specific types of embeddings, depending on which they
  * were trained on. For example, the default model `"ner_dl"` requires the
  * [[com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel WordEmbeddings]] `"glove_100d"`.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLSpec.scala NerDLSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
  * import org.apache.spark.ml.Pipeline
  *
  * // First extract the prerequisites for the NerDLModel
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("bert")
  *
  * // Then NER can be extracted
  * val nerTagger = NerDLModel.pretrained()
  *   .setInputCols("sentence", "token", "bert")
  *   .setOutputCol("ner")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   embeddings,
  *   nerTagger
  * ))
  *
  * val data = Seq("U.N. official Ekeus heads for Baghdad.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("ner.result").show(false)
  * +------------------------------------+
  * |result                              |
  * +------------------------------------+
  * |[B-ORG, O, O, B-PER, O, O, B-LOC, O]|
  * +------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel NerCrfModel]] for a generic CRF
  *   approach
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.ner.NerConverter NerConverter]] to further process the
  *   results
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
class NerDLModel(override val uid: String)
    extends AnnotatorModel[NerDLModel]
    with HasBatchedAnnotate[NerDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable
    with HasEngine {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  /** Input Annotator Types: DOCUMENT, TOKEN, WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)

  /** Output Annnotator type: NAMED_ENTITY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = NAMED_ENTITY

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group param
    */
  val minProba = new FloatParam(
    this,
    "minProbe",
    "Minimum probability. Used only if there is no CRF on top of LSTM layer.")

  /** datasetParams
    *
    * @group param
    */
  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Whether to include confidence scores in annotation metadata (Default: `false`)
    *
    * @group param
    */
  val includeConfidence = new BooleanParam(
    this,
    "includeConfidence",
    "Whether to include confidence scores in annotation metadata")

  /** whether to include all confidence scores in annotation metadata or just score of the
    * predicted tag
    *
    * @group param
    */
  val includeAllConfidenceScores = new BooleanParam(
    this,
    "includeAllConfidenceScores",
    "whether to include all confidence scores in annotation metadata")

  val classes =
    new StringArrayParam(this, "classes", "keep an internal copy of classes for Python")

  private var _model: Option[Broadcast[TensorflowNer]] = None

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group setParam
    */
  def setMinProbability(minProba: Float): this.type = set(this.minProba, minProba)

  /** datasetParams
    *
    * @group setParam
    */
  def setDatasetParams(params: DatasetEncoderParams): this.type = set(this.datasetParams, params)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): this.type = set(this.configProtoBytes, bytes)

  /** Whether to include confidence scores in annotation metadata
    *
    * @group setParam
    */
  def setIncludeConfidence(value: Boolean): this.type = set(this.includeConfidence, value)

  /** whether to include confidence scores for all tags rather than just for the predicted one
    *
    * @group setParam
    */
  def setIncludeAllConfidenceScores(value: Boolean): this.type =
    set(this.includeAllConfidenceScores, value)

  def setModelIfNotSet(spark: SparkSession, tf: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new NerDatasetEncoder(datasetParams.get.get)
      _model = Some(spark.sparkContext.broadcast(new TensorflowNer(tf, encoder, Verbose.Silent)))
    }
    this
  }

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group getParam
    */
  def getMinProba: Float = $(this.minProba)

  /** datasetParams
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group getParam
    */
  def getModelIfNotSet: TensorflowNer = _model.get.value

  /** Whether to include confidence scores in annotation metadata
    *
    * @group getParam
    */
  def getIncludeConfidence: Boolean = $(includeConfidence)

  /** whether to include all confidence scores in annotation metadata or just the score of the
    * predicted tag
    *
    * @group getParam
    */
  def getIncludeAllConfidenceScores: Boolean = $(includeAllConfidenceScores)

  /** get the tags used to trained this NerDLModel
    *
    * @group getParam
    */
  def getClasses: Array[String] = {
    val encoder = new NerDatasetEncoder(datasetParams.get.get)
    set(classes, encoder.tags)
    encoder.tags
  }

  setDefault(includeConfidence -> false, includeAllConfidenceScores -> false, batchSize -> 32)

  private case class RowIdentifiedSentence(
      rowIndex: Int,
      rowSentence: WordpieceEmbeddingsSentence)

  def tag(tokenized: Array[Array[WordpieceEmbeddingsSentence]]): Seq[Array[NerTaggedSentence]] = {
    val batch = tokenized.zipWithIndex.flatMap { case (t, i) =>
      t.map(RowIdentifiedSentence(i, _))
    }
    // Predict
    val labels = getModelIfNotSet.predict(
      batch.map(_.rowSentence),
      getConfigProtoBytes,
      includeConfidence = $(includeConfidence),
      includeAllConfidenceScores = $(includeAllConfidenceScores),
      $(batchSize))

    val outputBatches = Array.fill[Array[NerTaggedSentence]](tokenized.length)(Array.empty)

    // Combine labels with sentences tokens
    batch.indices.foreach { i =>
      val sentence = batch(i).rowSentence

      val tokens = sentence.tokens.indices.flatMap { j =>
        val token = sentence.tokens(j)
        val label = labels(i)(j)
        if (token.isWordStart) {
          Some(IndexedTaggedWord(token.token, label._1, token.begin, token.end, label._2))
        } else {
          None
        }
      }.toArray

      outputBatches(batch(i).rowIndex) =
        outputBatches(batch(i).rowIndex) :+ new TaggedSentence(tokens)
    }
    outputBatches
  }

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateStorageRef(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)
    dataset
  }

  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    // Parse
    val tokenized = batchedAnnotations
      .map(annotations => WordpieceEmbeddingsSentence.unpack(annotations).toArray)
      .toArray

    // Predict
    val tagged = tag(tokenized)

    // Pack
    tagged.map(innerTagged => NerTagged.pack(innerTagged))
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_nerdl",
      NerDLModel.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadsNERGraph extends ParamsAndFeaturesReadable[NerDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  def readNerGraph(instance: NerDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_nerdl")
    instance.setModelIfNotSet(spark: SparkSession, tf)
    // This allows for Python to access getClasses function
    val encoder = new NerDatasetEncoder(instance.datasetParams.get.get)
    instance.set(instance.classes, encoder.tags)
  }

  addReader(readNerGraph)
}

trait ReadablePretrainedNerDL
    extends ParamsAndFeaturesReadable[NerDLModel]
    with HasPretrained[NerDLModel] {
  override val defaultModelName: Some[String] = Some("ner_dl")

  override def pretrained(name: String, lang: String, remoteLoc: String): NerDLModel = {
    ResourceDownloader.downloadModel(NerDLModel, name, Option(lang), remoteLoc)
  }

  /** Java compliant-overrides */
  override def pretrained(): NerDLModel =
    pretrained(defaultModelName.get, defaultLang, defaultLoc)

  override def pretrained(name: String): NerDLModel = pretrained(name, defaultLang, defaultLoc)

  override def pretrained(name: String, lang: String): NerDLModel =
    pretrained(name, lang, defaultLoc)
}

/** This is the companion object of [[NerDLModel]]. Please refer to that class for the
  * documentation.
  */
object NerDLModel extends ReadablePretrainedNerDL with ReadsNERGraph
