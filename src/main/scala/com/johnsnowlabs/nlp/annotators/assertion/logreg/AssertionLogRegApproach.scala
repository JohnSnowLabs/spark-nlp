package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.embeddings.{AnnotatorWithWordEmbeddings, WordEmbeddings}
import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Dataset

/**
  * Created by jose on 22/11/17.
  */
class AssertionLogRegApproach(override val uid: String) extends
  AnnotatorWithWordEmbeddings[AssertionLogRegApproach, AssertionLogRegModel] with Windowing {

  override val requiredAnnotatorTypes = Array(DOCUMENT)
  override val description: String = "Clinical Text Status Assertion"
  override val tokenizer: Tokenizer = new SimpleTokenizer

  lazy override val (before, after) = (getOrDefault(beforeParam), getOrDefault(afterParam))

  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))
  //override lazy val localPath = getOrDefault(sourceEmbeddingsPath)

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val labelColumn = new Param[String](this, "label", "Column with one label per document")
  // the document where we're extracting the assertion
  val documentColumn = new Param[String](this, "document", "Column with one label per document")
  // the target term, that must appear capitalized in the document, e.g., 'diabetes'
  val targetColumn = new Param[String](this, "target", "Column with the target to analyze")
  val maxIter = new Param[Int](this, "maxIter", "Max number of iterations for algorithm")
  val regParam = new Param[Double](this, "regParam", "Regularization parameter")
  val eNetParam = new Param[Double](this, "eNetParam", "Elastic net parameter")
  val beforeParam = new Param[Int](this, "before", "Length of the context before the target")
  val afterParam = new Param[Int](this, "after", "Length of the context after the target")

  def setLabelCol(label: String) = set(labelColumn, label)
  def setDocumentCol(document: String) = set(documentColumn, document)
  def setTargetCol(target: String) = set(targetColumn, target)
  def setMaxIter(max: Int) = set(maxIter, max)
  def setReg(lambda: Double) = set(regParam, lambda)
  def setEnet(enet: Double) = set(eNetParam, enet)
  def setBefore(before: Int) = set(beforeParam, before)
  def setAfter(after: Int) = set(afterParam, after)

  setDefault(labelColumn -> "label",
    documentColumn -> "document",
    targetColumn   -> "target",
    maxIter -> 26,
    regParam -> 0.00192,
    eNetParam -> 0.9,
    beforeParam -> 10,
    afterParam -> 10
  )

  override def train(dataset: Dataset[_]): AssertionLogRegModel = {
    import dataset.sqlContext.implicits._

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.
      withColumn("features", applyWindowUdf($"text", $"target", $"start", $"end"))

    val lr = new LogisticRegression()
      .setMaxIter(getOrDefault(maxIter))
      .setRegParam(getOrDefault(regParam))
      .setElasticNetParam(getOrDefault(eNetParam))

    AssertionLogRegModel(lr.fit(processed))
  }

  override lazy val wordVectors: Option[WordEmbeddings] = embeddings

}
