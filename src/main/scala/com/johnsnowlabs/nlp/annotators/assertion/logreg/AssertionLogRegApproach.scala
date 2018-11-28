package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.embeddings.{ApproachWithWordEmbeddings, WordEmbeddingsRetriever}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.ml.param._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._

/**
  * Created by jose on 22/11/17.
  */
class AssertionLogRegApproach(val uid: String)
  extends ApproachWithWordEmbeddings[AssertionLogRegApproach, AssertionLogRegModel] with Windowing {

  override val requiredAnnotatorTypes = Array(DOCUMENT, CHUNK)
  val description: String = "Clinical Text Status Assertion"
  override val tokenizer: Tokenizer = new SimpleTokenizer
  override def wordVectors(): WordEmbeddingsRetriever = getClusterEmbeddings.getLocalRetriever

  lazy override val (before, after) = (getOrDefault(beforeParam), getOrDefault(afterParam))

  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val label = new Param[String](this, "label", "Column with one label per document")
  val maxIter = new IntParam(this, "maxIter", "Max number of iterations for algorithm")
  val regParam = new DoubleParam(this, "regParam", "Regularization parameter")
  val eNetParam = new DoubleParam(this, "eNetParam", "Elastic net parameter")
  val beforeParam = new IntParam(this, "beforeParam", "Amount of tokens from the context before the target")
  val afterParam = new IntParam(this, "afterParam", "Amount of tokens from the context after the target")
  val startCol = new Param[String](this, "startCol", "Column that contains the token number for the start of the target")
  val endCol = new Param[String](this, "endCol", "Column that contains the token number for the end of the target")

  def setLabelCol(label: String): this.type = set(label, label)
  def setMaxIter(max: Int): this.type = set(maxIter, max)
  def setReg(lambda: Double): this.type = set(regParam, lambda)
  def setEnet(enet: Double): this.type = set(eNetParam, enet)
  def setBefore(b: Int): this.type = set(beforeParam, b)
  def setAfter(a: Int): this.type = set(afterParam, a)
  def setStartCol(start: String): this.type = set(startCol, start)
  def setEndCol(end: String): this.type = set(endCol, end)

  setDefault(label -> "label",
    maxIter -> 26,
    regParam -> 0.00192,
    eNetParam -> 0.9,
    beforeParam -> 10,
    afterParam -> 10
  )

  private def processWithChunk(dataset: DataFrame): DataFrame = {
    val documentCol = dataset.schema.fields
      .find(f => $(inputCols).contains(f.name) && f.metadata.getString("annotatorType") == DOCUMENT)
      .get.name
    val chunkCol = dataset.schema.fields
      .find(f => $(inputCols).contains(f.name) && f.metadata.getString("annotatorType") == CHUNK)
      .get.name


    dataset.toDF
      .withColumn("_features",
        /** explode will delete rows that do not contain any chunk. Will only train chunked rows.
          * Transform will explode_outer instead */
        explode(applyWindowUdfChunk(col(documentCol), col(chunkCol)))
      )
  }

  private def processWithStartEnd(dataset: DataFrame): DataFrame = {
    val documentCol = dataset.schema.fields
      .find(f => $(inputCols).contains(f.name) && f.metadata.getString("annotatorType") == DOCUMENT)
      .get.name

    dataset.toDF
      .withColumn("_features",
          applyWindowUdf(col(documentCol),
          col($(startCol)),
          col($(endCol)))
      )
  }


  private def trainWithChunk(dataset: Dataset[_], labelCol: String, labelMappings: Map[String, Double]): DataFrame = {

    val preprocessed = dataset
      .withColumn(labelCol, labelToNumber(labelMappings)(col(labelCol)))

    processWithChunk(preprocessed)
  }

  private def trainWithStartEnd(dataset: Dataset[_], labelCol: String, labelMappings: Map[String, Double]): DataFrame = {

    val preprocessed = dataset
      .withColumn(labelCol, labelToNumber(labelMappings)(col(labelCol)))

    processWithStartEnd(preprocessed)
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel] = None): AssertionLogRegModel = {

    // Preload word embeddings before doing it within udf
    wordVectors()

    val lr = new LogisticRegression()
      .setMaxIter(getOrDefault(maxIter))
      .setRegParam(getOrDefault(regParam))
      .setElasticNetParam(getOrDefault(eNetParam))
      .setPredictionCol("_prediction")
      .setFeaturesCol("_features")

    val labelCol = getOrDefault(label)

    /* infer labels and assign a number to each */
    val labelMappings: Map[String, Double] = dataset.select(labelCol).distinct.collect
      .map(row => row.getAs[String](labelCol)).zipWithIndex
      .map{case (labelK, idx) => (labelK, idx.toDouble)}
      .toMap

    /* apply UDF to fix the length of each document */
    val processed =
    if (get(startCol).isDefined & get(endCol).isDefined) {
      trainWithStartEnd(dataset, labelCol, labelMappings)
    } else {
      trainWithChunk(dataset, labelCol, labelMappings)
    }

    new AssertionLogRegModel()
      .setBefore(getOrDefault(beforeParam))
      .setAfter(getOrDefault(afterParam))
      .setInputCols(getOrDefault(inputCols))
      .setLabelMap(labelMappings)
      .setModel(lr.fit(processed))
  }

  private def labelToNumber(mappings: Map[String, Double]) = udf { label:String  => mappings.get(label)}

}

object AssertionLogRegApproach extends DefaultParamsReadable[AssertionLogRegApproach]