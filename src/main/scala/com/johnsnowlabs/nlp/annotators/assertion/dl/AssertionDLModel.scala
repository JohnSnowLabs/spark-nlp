package com.johnsnowlabs.nlp.annotators.assertion.dl

import java.io.File
import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.ParamMap
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
  with ParamsAndFeaturesWritable
  with TransformModelSchema {

  def this() = this(Identifiable.randomUID("ASSERTION"))

  def setDatasetParams(params: DatasetEncoderParams): AssertionDLModel = set(this.datasetParams, params)

  var tensorflow: TensorflowWrapper = null

  def setTensorflow(tf: TensorflowWrapper): AssertionDLModel = {
    this.tensorflow = tf
    this
  }

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
        10,
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
      // TODO single datapoint version, use mapPartitions instead
      withColumn(getOutputCol, packAnnotations(col("text"), col("start"), col("end"))
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

  /* send this to common place */
  def extractTextUdf: UserDefinedFunction = udf { document:mutable.WrappedArray[GenericRowWithSchema] =>
     document.head.getString(3)
  }


  override val requiredAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val annotatorType: AnnotatorType = ASSERTION


  /** requirement for annotators copies */
  override def copy(extra: ParamMap): AssertionDLModel = defaultCopy(extra)

  /* TODO Refactor put common code in some other place */
  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_assertiondl")
      .toAbsolutePath.toString
    val tfFile = Paths.get(tmpFolder, AssertionDLModel.tfFile).toString

    // 2. Save Tensorflow state
    tensorflow.saveToFile(tfFile)

    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(tfFile), new Path(path))

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }

}

/* TODO Refactor put common code in some other place */
object AssertionDLModel extends ParamsAndFeaturesReadable[AssertionDLModel] {

  val tfFile = "tensorflow"

  override def onRead(instance: AssertionDLModel, path: String, spark: SparkSession): Unit = {

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_assertiondl")
      .toAbsolutePath.toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, tfFile), new Path(tmpFolder))

    // 3. Read Tensorflow state
    val tf = TensorflowWrapper.read(new Path(tmpFolder, tfFile).toString)
    instance.setTensorflow(tf)

    // 4. Remove tmp folder
    FileUtils.deleteDirectory(new File(tmpFolder))
  }
}