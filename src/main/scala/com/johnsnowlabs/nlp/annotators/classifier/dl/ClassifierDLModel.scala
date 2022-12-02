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
import org.apache.spark.ml.param.{IntArrayParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}

/** ClassifierDL for generic Multi-class Text Classification.
  *
  * ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text
  * classifications. The ClassifierDL annotator uses a deep learning model (DNNs) we have built
  * inside TensorFlow and supports up to 100 classes.
  *
  * This is the instantiated model of the [[ClassifierDLApproach]]. For training your own model,
  * please see the documentation of that class.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val classifierDL = ClassifierDLModel.pretrained()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCol("classification")
  * }}}
  * The default model is `"classifierdl_use_trec6"`, if no name is provided. It uses embeddings
  * from the [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]]
  * and is trained on the
  * [[https://deepai.org/dataset/trec-6#:~:text=The%20TREC%20dataset%20is%20dataset,50%20has%20finer%2Dgrained%20labels TREC-6]]
  * dataset. For available pretrained models please see the
  * [[https://nlp.johnsnowlabs.com/models?task=Text+Classification Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala ClassifierDLTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotator.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLModel
  * import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val useEmbeddings = UniversalSentenceEncoder.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("sentence_embeddings")
  *
  * val sarcasmDL = ClassifierDLModel.pretrained("classifierdl_use_sarcasm")
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCol("sarcasm")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     sentence,
  *     useEmbeddings,
  *     sarcasmDL
  *   ))
  *
  * val data = Seq(
  *   "I'm ready!",
  *   "If I could put into words how much I love waking up at 6 am on Mondays I would."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(arrays_zip(sentence, sarcasm)) as out")
  *   .selectExpr("out.sentence.result as sentence", "out.sarcasm.result as sarcasm")
  *   .show(false)
  * +-------------------------------------------------------------------------------+-------+
  * |sentence                                                                       |sarcasm|
  * +-------------------------------------------------------------------------------+-------+
  * |I'm ready!                                                                     |normal |
  * |If I could put into words how much I love waking up at 6 am on Mondays I would.|sarcasm|
  * +-------------------------------------------------------------------------------+-------+
  * }}}
  *
  * @see
  *   [[MultiClassifierDLModel]] for multi-class classification
  * @see
  *   [[SentimentDLModel]] for sentiment analysis
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
class ClassifierDLModel(override val uid: String)
    extends AnnotatorModel[ClassifierDLModel]
    with HasSimpleAnnotate[ClassifierDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable
    with HasEngine {
  def this() = this(Identifiable.randomUID("ClassifierDLModel"))

  /** Output annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  /** Output annotator type : CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = CATEGORY

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * config_proto.SerializeToString()
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group setParam
    */
  def setConfigProtoBytes(bytes: Array[Int]): ClassifierDLModel.this.type =
    set(this.configProtoBytes, bytes)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group getParam
    */
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /** Dataset params
    *
    * @group param
    */
  val datasetParams = new StructFeature[ClassifierDatasetEncoderParams](this, "datasetParams")

  /** Labels used to train this model
    *
    * @group param
    */
  val classes = new StringArrayParam(this, "classes", "Labels used to train this model")

  /** Dataset params
    *
    * @group setParam
    */
  def setDatasetParams(params: ClassifierDatasetEncoderParams): ClassifierDLModel.this.type =
    set(this.datasetParams, params)

  private var _model: Option[Broadcast[TensorflowClassifier]] = None

  def setModelIfNotSet(spark: SparkSession, tf: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)

      _model = Some(
        spark.sparkContext.broadcast(new TensorflowClassifier(tf, encoder, None, Verbose.Silent)))
    }
    this
  }

  def getModelIfNotSet: TensorflowClassifier = _model.get.value

  /** Labels used to train this model
    *
    * @group getParam
    */
  def getClasses: Array[String] = {
    val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)
    set(classes, encoder.tags)
    encoder.tags
  }

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
      getModelIfNotSet.predict(sentences, getConfigProtoBytes)
    else Seq.empty[Annotation]

  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_classifierdl",
      ClassifierDLModel.tfFile,
      configProtoBytes = getConfigProtoBytes)

  }
}

trait ReadablePretrainedClassifierDL
    extends ParamsAndFeaturesReadable[ClassifierDLModel]
    with HasPretrained[ClassifierDLModel] {
  override val defaultModelName: Some[String] = Some("classifierdl_use_trec6")

  override def pretrained(name: String, lang: String, remoteLoc: String): ClassifierDLModel = {
    ResourceDownloader.downloadModel(ClassifierDLModel, name, Option(lang), remoteLoc)
  }

  /** Java compliant-overrides */
  override def pretrained(): ClassifierDLModel =
    pretrained(defaultModelName.get, defaultLang, defaultLoc)

  override def pretrained(name: String): ClassifierDLModel =
    pretrained(name, defaultLang, defaultLoc)

  override def pretrained(name: String, lang: String): ClassifierDLModel =
    pretrained(name, lang, defaultLoc)
}

trait ReadClassifierDLTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[ClassifierDLModel] =>

  override val tfFile: String = "classifierdl_tensorflow"

  def readTensorflow(instance: ClassifierDLModel, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_classifierdl_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
    // This allows for Python to access getClasses function
    val encoder = new ClassifierDatasetEncoder(instance.datasetParams.get.get)
    instance.set(instance.classes, encoder.tags)
  }

  addReader(readTensorflow)
}

/** This is the companion object of [[ClassifierDLModel]]. Please refer to that class for the
  * documentation.
  */
object ClassifierDLModel
    extends ReadablePretrainedClassifierDL
    with ReadClassifierDLTensorflowModel
