package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegModel
import com.johnsnowlabs.nlp.embeddings.EmbeddingsReadable
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{IntParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

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
  val targetNerLabels = new StringArrayParam(this, "targetNerLabels", "List of NER labels to mark as target for assertion, must match NER output")

  def setStartCol(s: String): this.type = set(startCol, s)
  def setEndCol(e: String): this.type = set(endCol, e)
  def setNerCol(col: String): this.type = set(nerCol, col)
  def setTargetNerLabels(v: Array[String]): this.type = set(targetNerLabels, v)

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

  private def generateEmptyAnnotations = udf {
    () => Seq.empty[Annotation]
  }

  override final def transform(dataset: Dataset[_]): DataFrame = {
    require(validate(dataset.schema), s"Missing annotators in pipeline. Make sure the following are present: " +
      s"${requiredAnnotatorTypes.mkString(", ")}")

    /* apply UDFs to classify and annotate */
    val prefiltered = if (get(nerCol).isDefined) {
      require(get(targetNerLabels).isDefined, "Param targetNerLabels must be defined in order to use NER based assertion status")
      dataset.toDF.
        filter(r => {
          val annotations = r.getAs[Seq[Row]]($(nerCol)).map(Annotation(_))
          annotations.exists(a => $(targetNerLabels).contains(a.result))
        })
      } else {
        dataset.toDF
      }

    val packed = prefiltered.
      withColumn("_text", extractTextUdf(col(getInputCols.head))).
      withColumn(getOutputCol, {
        if (get(nerCol).isDefined) {
          require(get(targetNerLabels).isDefined, "Param targetNerLabels must be defined in order to use NER based assertion status")
          packAnnotationsNer($(targetNerLabels))(col("_text"), col($(nerCol)))
        } else if (get(startCol).isDefined & get(endCol).isDefined) {
          packAnnotations(col("_text"), col($(startCol)), col($(endCol)))
        } else {
          throw new IllegalArgumentException("Either nerCol or startCol and endCol must be defined in order to predict assertion")
        }
      }).
      drop("_text")

    if (get(nerCol).isDefined) {
      packed.union(dataset.toDF.
        filter(r => {
          val annotations = r.getAs[Seq[Row]]($(nerCol)).map(Annotation(_))
          annotations.forall(a => !$(targetNerLabels).contains(a.result))
        }).withColumn(getOutputCol, generateEmptyAnnotations()))
    } else {
      packed
    }

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

  private def packAnnotationsNer(targetLabels: Array[String]) = udf { (text: String, n: Seq[Row]) =>
    val tokens = text.split(" ").filter(_!="")
    val annotations = n.map { r => Annotation(r) }
    val targets = annotations.zipWithIndex.filter(a => targetLabels.contains(a._1.result)).toIterator
    val ranges = ArrayBuffer.empty[(Int, Int)]
    while (targets.hasNext) {
      val annotation = targets.next
      var range = (annotation._1.begin, annotation._1.end)
      var look = true
      while(look && targets.hasNext) {
        val nextAnnotation = targets.next
        if (nextAnnotation._2 == annotation._2 + 1)
          range = (range._1, nextAnnotation._1.end)
        else
          look = false
      }
      ranges.append(range)
    }
    if (ranges.nonEmpty) {
      ranges.map {r => {
        val prediction = model.predict(Array(tokens), Array(r._1), Array(r._2)).head
        Annotation("assertion", r._1, r._2, prediction, Map())
      }}
    }
    else
      throw new IllegalArgumentException("NER Based assertion status failed due to missing entities in nerCol")
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
  def pretrained(name: String = "as_fast_dl", language: Option[String] = Some("en"), folder: String = ResourceDownloader.publicLoc): AssertionDLModel =
    ResourceDownloader.downloadModel(AssertionDLModel, name, language, folder)
}

object AssertionDLModel extends EmbeddingsReadable[AssertionDLModel] with ReadsAssertionGraph with PretrainedDLAssertionStatus