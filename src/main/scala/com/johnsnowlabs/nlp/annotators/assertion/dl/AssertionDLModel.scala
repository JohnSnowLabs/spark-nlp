package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel
import com.johnsnowlabs.nlp.embeddings.EmbeddingsReadable
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
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


  val nerCol = new Param[String](this, "nerCol", "Column with NER output annotation to find target token")
  val startCol = new Param[String](this, "startCol", "Column with token number for first target token")
  val endCol = new Param[String](this, "endCol", "Column with token number for last target token")

  def setStartCol(s: String): this.type = set(startCol, s)
  def setEndCol(e: String): this.type = set(endCol, e)
  def setNerCol(col: String): this.type = set(nerCol, col)

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

    /* apply UDFs to classify and annotate */
    dataset.toDF.
      withColumn("_text", extractTextUdf(col(getInputCols.head))).
      withColumn(getOutputCol, {
        if (get(nerCol).isDefined) {
          packAnnotationsNer(col("_text"), col($(nerCol)))
        } else if (get(startCol).isDefined & get(endCol).isDefined) {
          packAnnotations(col("_text"), col($(startCol)), col($(endCol)))
        } else {
          throw new IllegalArgumentException("Either nerCol or startCol and endCol must be defined in order to predict assertion")
        }
      }
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

  private def packAnnotationsNer = udf { (text: String, n: Seq[Row]) =>
    val tokens = text.split(" ").filter(_!="")
    n.flatMap{ nn => {
      val annotation = Annotation(nn)
      val prediction = model.predict(Array(tokens), Array(annotation.begin), Array(annotation.end)).head
      val resultAnnotation = Annotation("assertion", annotation.begin, annotation.end, prediction, Map())
      Seq(resultAnnotation)
    }}
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

  def readAssertionGraph(instance: AssertionDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_assertiondl")
    instance.setTensorflow(tf)
  }

  addReader(readAssertionGraph)
}

trait PretrainedDLAssertionStatus {
  def pretrained(name: String = "as_fast_dl", language: Option[String] = Some("en"), folder: String = ResourceDownloader.publicFolder): AssertionDLModel =
    ResourceDownloader.downloadModel(AssertionDLModel, name, language, folder)
}

object AssertionDLModel extends EmbeddingsReadable[AssertionDLModel] with ReadsAssertionGraph with PretrainedDLAssertionStatus