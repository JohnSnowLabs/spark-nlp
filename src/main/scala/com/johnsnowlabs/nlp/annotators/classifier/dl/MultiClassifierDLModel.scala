package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow.{ClassifierDatasetEncoder, ClassifierDatasetEncoderParams, ReadTensorflowModel, TensorflowMultiClassifier, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{FloatParam, IntArrayParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}


/**
  * MultiClassifierDL is a Multi-label Text Classification. MultiClassifierDL Bidirectional GRU with Convolution model we have built inside TensorFlow and supports up to 100 classes.
  * The input to MultiClassifierDL is Sentence Embeddings such as state-of-the-art UniversalSentenceEncoder, BertSentenceEmbeddings, or SentenceEmbeddings
  *
  * In machine learning, multi-label classification and the strongly related problem of multi-output classification are variants of the classification problem where multiple labels may be assigned to each instance. Multi-label classification is a generalization of multiclass classification, which is the single-label problem of categorizing instances into precisely one of more than two classes; in the multi-label problem there is no constraint on how many of the classes the instance can be assigned to.
  * Formally, multi-label classification is the problem of finding a model that maps inputs x to binary vectors y (assigning a value of 0 or 1 for each element (label) in y).
  * https://en.wikipedia.org/wiki/Multi-label_classification
  *
  * NOTE: This annotator accepts an array of labels in type of String.
  * NOTE: UniversalSentenceEncoder and SentenceEmbeddings can be used for the inputCol
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLTestSpec.scala]] for further reference on how to use this API
  *
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  *
  **/
class MultiClassifierDLModel(override val uid: String)
  extends AnnotatorModel[MultiClassifierDLModel] with HasSimpleAnnotate[MultiClassifierDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable {
  def this() = this(Identifiable.randomUID("MultiClassifierDLModel"))

  /** Output annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  /** Output annotator type : CATEGORY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = CATEGORY

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** The minimum threshold for each label to be accepted. Default is 0.5
    *
    * @group param
    **/
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for each label to be accepted. Default is 0.5")

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group setParam
    **/
  def setConfigProtoBytes(
                           bytes: Array[Int]
                         ): MultiClassifierDLModel.this.type = set(this.configProtoBytes, bytes)

  /** Tensorflow config Protobytes passed to the TF session
    *
    * @group getParam
    **/
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  /**
    * datasetParams
    *
    * @group param */
  val datasetParams = new StructFeature[ClassifierDatasetEncoderParams](this, "datasetParams")

  val classes = new StringArrayParam(this, "classes", "keep an internal copy of classes for Python")

  /**
    * datasetParams
    *
    * @group setParam */
  def setDatasetParams(params: ClassifierDatasetEncoderParams): MultiClassifierDLModel.this.type =
    set(this.datasetParams, params)

  def getClasses: Array[String] = {
    val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)
    set(classes, encoder.tags)
    encoder.tags
  }
  /** The minimum threshold for each label to be accepted. Default is 0.5
    *
    * @group setParam
    **/
  def setThreshold(threshold: Float): MultiClassifierDLModel.this.type = set(this.threshold, threshold)

  /** @group param */
  private var _model: Option[Broadcast[TensorflowMultiClassifier]] = None

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tf: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new ClassifierDatasetEncoder(datasetParams.get.get)

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowMultiClassifier(
            tf,
            encoder,
            Verbose.Silent
          )
        )
      )
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: TensorflowMultiClassifier = _model.get.value

  /** The minimum threshold for each label to be accepted. Default is 0.5
    *
    * @group getParam
    **/
  def getThreshold: Float = $(this.threshold)

  setDefault(
    threshold -> 0.5f
  )

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateStorageRef(dataset, $(inputCols), SENTENCE_EMBEDDINGS)
    dataset
  }

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = annotations
      .filter(_.annotatorType == SENTENCE_EMBEDDINGS)
      .groupBy(_.metadata.getOrElse[String]("sentence", "0").toInt)
      .toSeq
      .sortBy(_._1)

    val embeddingsLength = sentences.flatMap(x=>x._2.flatten(x=>x.embeddings)).nonEmpty

    if(embeddingsLength) {
      getModelIfNotSet.predict(sentences, $(threshold), getConfigProtoBytes)
    }else {
      Seq.empty[Annotation]
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_multiclassifierdl",
      MultiClassifierDLModel.tfFile,
      configProtoBytes = getConfigProtoBytes
    )

  }
}

trait ReadablePretrainedMultiClassifierDL
  extends ParamsAndFeaturesReadable[MultiClassifierDLModel]
    with HasPretrained[MultiClassifierDLModel] {
  override val defaultModelName: Some[String] = Some("multiclassifierdl_use_toxic")

  override def pretrained(name: String, lang: String, remoteLoc: String): MultiClassifierDLModel = {
    ResourceDownloader.downloadModel(MultiClassifierDLModel, name, Option(lang), remoteLoc)
  }

  /** Java compliant-overrides */
  override def pretrained(): MultiClassifierDLModel = pretrained(defaultModelName.get, defaultLang, defaultLoc)
  override def pretrained(name: String): MultiClassifierDLModel = pretrained(name, defaultLang, defaultLoc)
  override def pretrained(name: String, lang: String): MultiClassifierDLModel = pretrained(name, lang, defaultLoc)
}

trait ReadMultiClassifierDLTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[MultiClassifierDLModel] =>

  override val tfFile: String = "multiclassifierdl_tensorflow"

  def readTensorflow(instance: MultiClassifierDLModel, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_multiclassifierdl_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
    // This allows for Python to access getClasses function
    val encoder = new ClassifierDatasetEncoder(instance.datasetParams.get.get)
    instance.set(instance.classes, encoder.tags)
  }

  addReader(readTensorflow)
}

object MultiClassifierDLModel extends ReadablePretrainedMultiClassifierDL with ReadMultiClassifierDLTensorflowModel
