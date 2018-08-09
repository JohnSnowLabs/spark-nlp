package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.embeddings.{EmbeddingsReadable, WordEmbeddings}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.param._
import org.apache.spark.sql.functions._

import scala.collection.immutable.Map

/**
  * Created by jose on 22/11/17.
  */

class AssertionLogRegModel(override val uid: String) extends RawAnnotator[AssertionLogRegModel]
  with Windowing with HasWordEmbeddings  {

  override val tokenizer: Tokenizer = new SimpleTokenizer
  override val annotatorType: AnnotatorType = ASSERTION
  override val requiredAnnotatorTypes = Array(DOCUMENT, CHUNK)
  override lazy val wordVectors: Option[WordEmbeddings] = embeddings

  val beforeParam = new IntParam(this, "beforeParam", "Length of the context before the target")
  val afterParam = new IntParam(this, "afterParam", "Length of the context after the target")

  val startCol = new Param[String](this, "startCol", "Column that contains the token number for the start of the target")
  val endCol = new Param[String](this, "endCol", "Column that contains the token number for the end of the target")

  var model: StructFeature[LogisticRegressionModel] = new StructFeature[LogisticRegressionModel](this, "logistic regression")
  var labelMap: MapFeature[Double, String] = new MapFeature[Double, String](this, "labels")

  override lazy val (before, after) = (getOrDefault(beforeParam), getOrDefault(afterParam))

  setDefault(
    beforeParam -> 11,
    afterParam -> 13
  )

  def this() = this(Identifiable.randomUID("ASSERTION"))

  def setBefore(before: Int): this.type = set(beforeParam, before)
  def setAfter(after: Int): this.type = set(afterParam, after)
  def setStartCol(start: String): this.type = set(startCol, start)
  def setEndCol(end: String): this.type = set(endCol, end)

  override final def transform(dataset: Dataset[_]): DataFrame = {
    require(validate(dataset.schema), s"Missing annotators in pipeline. Make sure the following are present: " +
      s"${requiredAnnotatorTypes.mkString(", ")}")

    import dataset.sqlContext.implicits._

    val documentCol = dataset.schema.fields
      .find(f => $(inputCols).contains(f.name) && f.metadata.getString("annotatorType") == DOCUMENT)
      .get.name
    val chunkCol = dataset.schema.fields
      .find(f => $(inputCols).contains(f.name) && f.metadata.getString("annotatorType") == CHUNK)
      .get.name

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.
      withColumn("_rid", monotonically_increasing_id()).
      /** explode_outer will nullify non-chunked rows */
      withColumn("_features", explode_outer(applyWindowUdfChunk(col(documentCol), col(chunkCol))))

    val resultData = $$(model).transform(processed).withColumn("_tmpassertion", {
      packAnnotationsFromChunks(col("_features"), $"_prediction")
    }).drop("_prediction", "_features", "rawPrediction", "probability")

    val packedData = {
      val firstOfAll = resultData.drop("_rid").columns.map(c => first(col(c)).as(c))
      resultData
        .groupBy("_rid")
        .agg(collect_list(col("_tmpassertion")).as(getOutputCol), firstOfAll:_*)
        .drop("_rid", "_tmpassertion")
    }

    packedData

  }

  private def packAnnotationsFromChunks = udf { (vector: org.apache.spark.ml.linalg.Vector, prediction: Double) =>
    if (vector.numNonzeros > 0)
     Annotation("assertion", vector.apply(1).toInt, vector.apply(2).toInt, $$(labelMap)(prediction), Map())
    else
      Annotation("assertion", vector.apply(1).toInt, vector.apply(2).toInt, "NA", Map())
  }

  def setModel(m: LogisticRegressionModel): this.type = set(model, m)

  def setLabelMap(labelMappings: Map[String, Double]): this.type = set(labelMap, labelMappings.map(_.swap))

  /** requirement for annotators copies */
  override def copy(extra: ParamMap): AssertionLogRegModel = defaultCopy(extra)
}

trait PretrainedAssertionLogRegModel {
  def pretrained(name: String = "as_fast_lg", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): AssertionLogRegModel =
    ResourceDownloader.downloadModel(AssertionLogRegModel, name, language, remoteLoc)
}


object AssertionLogRegModel extends EmbeddingsReadable[AssertionLogRegModel] with PretrainedAssertionLogRegModel
