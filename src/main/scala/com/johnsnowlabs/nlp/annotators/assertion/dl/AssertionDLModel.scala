package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

import scala.collection.mutable

/**
  * Created by jose on 14/03/18.
  */
class AssertionDLModel(override val uid: String) extends RawAnnotator[AssertionDLModel]
  with HasWordEmbeddings
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable
  with TransformModelSchema {


  val start = new Param[String](this, "start", "Column with token number for first target token")
  val end = new Param[String](this, "end", "Column with token number for last target token")

  def setStart(s: String): this.type = set(start, s)
  def setEnd(e: String): this.type = set(end, e)

  def this() = this(Identifiable.randomUID("ASSERTION"))

  def setDatasetParams(params: DatasetEncoderParams): AssertionDLModel = set(this.datasetParams, params)

  var tensorflow: TensorflowWrapper = null

  def setTensorflow(tf: TensorflowWrapper): AssertionDLModel = {
    this.tensorflow = tf
    this
  }

  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  def setBatchSize(size: Int) = set(this.batchSize, size)

  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")

  @transient
  private var _model: TensorflowAssertion = null

  def model: TensorflowAssertion = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")
      require(embeddings.isDefined, "Embeddings must be defined before usage")
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new AssertionDatasetEncoder(embeddings.get.getEmbeddings, datasetParams.get.get)
      _model = new TensorflowAssertion(
        tensorflow,
        encoder,
        ${batchSize},
        Verbose.Silent)
    }

    _model
  }

  override final def transform(dataset: Dataset[_]): DataFrame = {
    require(validate(dataset.schema), s"Missing annotators in pipeline. Make sure the following are present: " +
      s"${requiredAnnotatorTypes.mkString(", ")}")

    import dataset.sqlContext.implicits._

    /* apply UDFs to classify and annotate */
    dataset.toDF.
      withColumn("text", extractTextUdf(col(getInputCols.head))).
      withColumn(getOutputCol, packAnnotations(col("text"), col(getOrDefault(start)), col(getOrDefault(end)))
    )
  }

  private def packAnnotations = udf { (text: String, s: Int, e: Int) =>
    val tokens = text.split(" ").filter(_!="")

    /* convert from token indices in s,e to indices in the doc string */
    val start = tokens.slice(0, s).map(_.length).sum +
      tokens.slice(0, s).length // account for spaces
    val end = start + tokens.slice(s, e + 1).map(_.length).sum +
      tokens.slice(s, e + 1).length - 2 // account for spaces

    val prediction = model.predict(Array(tokens), Array(s), Array(e)).head
    val annotation = Annotation("assertion", start, end, prediction, Map())
    Seq(annotation)
  }

  def extractTextUdf: UserDefinedFunction = udf { document:mutable.WrappedArray[GenericRowWithSchema] =>
     document.head.getAs[String]("result")
  }

  override val requiredAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val annotatorType: AnnotatorType = ASSERTION

  /** requirement for annotators copies */
  override def copy(extra: ParamMap): AssertionDLModel = defaultCopy(extra)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, tensorflow, "_assertiondl")
  }

}

trait ReadsAssertionGraph extends ParamsAndFeaturesReadable[AssertionDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  override def onRead(instance: AssertionDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_assertiondl")
    instance.setTensorflow(tf)
  }
}

object AssertionDLModel extends ParamsAndFeaturesReadable[AssertionDLModel] with ReadsAssertionGraph