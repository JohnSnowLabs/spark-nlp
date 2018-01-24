package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.embeddings.{AnnotatorWithWordEmbeddings, WordEmbeddings}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

import scala.collection.mutable

/**
  * Created by jose on 22/11/17.
  */
class AssertionLogRegApproach(override val uid: String) extends AnnotatorApproach[AssertionLogRegModel] with
  AnnotatorWithWordEmbeddings[AssertionLogRegApproach, AssertionLogRegModel] with Windowing {

  override val requiredAnnotatorTypes = Array(DOCUMENT)
  override val description: String = "Clinical Text Status Assertion"
  override val tokenizer: Tokenizer = new SimpleTokenizer
  override lazy val wordVectors: Option[WordEmbeddings] = embeddings

  lazy override val (before, after) = (getOrDefault(beforeParam), getOrDefault(afterParam))

  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val label = new Param[String](this, "label", "Column with one label per document")
  // the document where we're extracting the assertion
  val document = new Param[String](this, "document", "Column with the text to be analyzed")
  // the target term, that must appear capitalized in the document, e.g., 'diabetes'
  val target = new Param[String](this, "target", "Column with the target to analyze")
  val maxIter = new Param[Int](this, "maxIter", "Max number of iterations for algorithm")
  val regParam = new Param[Double](this, "regParam", "Regularization parameter")
  val eNetParam = new Param[Double](this, "eNetParam", "Elastic net parameter")
  val beforeParam = new IntParam(this, "beforeParam", "Length of the context before the target")
  val afterParam = new IntParam(this, "afterParam", "Length of the context after the target")

  def setLabelCol(label: String) = set(label, label)
  def setDocumentCol(document: String) = set(document, document)
  def setTargetCol(target: String) = set(target, target)
  def setMaxIter(max: Int) = set(maxIter, max)
  def setReg(lambda: Double) = set(regParam, lambda)
  def setEnet(enet: Double) = set(eNetParam, enet)
  def setBefore(b: Int) = set(beforeParam, b)
  def setAfter(a: Int) = set(afterParam, a)

  setDefault(label -> "label",
    document -> "document",
    target   -> "target",
    maxIter -> 26,
    regParam -> 0.00192,
    eNetParam -> 0.9,
    beforeParam -> 10,
    afterParam -> 10
  )

  /* send this to common place */
  def extractTextUdf = udf { document:mutable.WrappedArray[GenericRowWithSchema] =>
      document.head.getString(3)
  }

  override def train(dataset: Dataset[_]): AssertionLogRegModel = {
    import dataset.sqlContext.implicits._

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.
      withColumn("text", extractTextUdf($"document")).
      withColumn("features", applyWindowUdf($"text", $"target", $"start", $"end"))

    val lr = new LogisticRegression()
      .setMaxIter(getOrDefault(maxIter))
      .setRegParam(getOrDefault(regParam))
      .setElasticNetParam(getOrDefault(eNetParam))

    val labelCol = getOrDefault(label)

    /* infer labels and assign a number to each */
    val labelMappings: Map[String, Double] = dataset.select(labelCol).distinct.collect
        .map(row => row.getAs[String](labelCol)).zipWithIndex
        .map{case (label, idx) => (label, idx.toDouble)}
        .toMap

    val processedWithLabel = processed.withColumn(labelCol, labelToNumber(labelMappings)(col(labelCol)))

    AssertionLogRegModel()
      .setBefore(getOrDefault(beforeParam))
      .setAfter(getOrDefault(afterParam))
      .setLabelMap(labelMappings)
      .setModel(lr.fit(processedWithLabel))
  }

  private def labelToNumber(mappings: Map[String, Double]) = udf { label:String  => mappings.get(label)}
}
