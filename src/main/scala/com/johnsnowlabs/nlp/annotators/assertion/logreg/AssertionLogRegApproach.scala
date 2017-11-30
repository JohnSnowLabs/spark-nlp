package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.embeddings.AnnotatorWithWordEmbeddings
import com.johnsnowlabs.nlp.{AnnotatorApproach}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.classification.LogisticRegression

/**
  * Created by jose on 22/11/17.
  */
class AssertionLogRegApproach extends AnnotatorApproach[AssertionLogRegModel]
  with AnnotatorWithWordEmbeddings with Windowing {

  override val requiredAnnotatorTypes = Array(TOKEN)
  override val description: String = "Clinical Text Status Assertion"
  override val annotatorType: AnnotatorType = null
  override val uid: String = ""

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val labelColumn = new Param[String](this, "labelColumn", "Column with one label per document")

  // the document where we're extracting the assertion
  val documentColumn = new Param[String](this, "documentColumn", "Column with one label per document")

  // the target term, that must appear capitalized in the document, e.g., 'diabetes'
  val targetColumn = new Param[String](this, "targetColumn", "Column with the target to analyze")

  override val (before, after) = (10, 5)

  override def train(dataset: Dataset[_]): AssertionLogRegModel = {
    import dataset.sqlContext.implicits._

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.select(applyWindowUdf($"documentColumn", $"targetColumn")
      .as("window"), $"labelColumn".as("label"))

    /* TODO: pick the parameters you want to expose*/
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    lr.fit(processed)

    AssertionLogRegModel(lr)
  }

}
