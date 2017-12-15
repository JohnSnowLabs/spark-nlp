package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.embeddings.AnnotatorWithWordEmbeddings
import com.johnsnowlabs.nlp.AnnotatorApproach
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.Param

/**
  * Created by jose on 22/11/17.
  */
class AssertionLogRegApproach(override val uid: String) extends AnnotatorApproach[AssertionLogRegModel]
  with AnnotatorWithWordEmbeddings with Windowing {

  override val requiredAnnotatorTypes = Array(DOCUMENT, POS)
  override val description: String = "Clinical Text Status Assertion"
  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))
  override lazy  val localPath = "/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin.db"

  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val labelColumn = new Param[String](this, "label", "Column with one label per document")

  // the document where we're extracting the assertion
  val documentColumn = new Param[String](this, "documentColumn", "Column with one label per document")

  // the target term, that must appear capitalized in the document, e.g., 'diabetes'
  val targetColumn = new Param[String](this, "targetColumn", "Column with the target to analyze")

  override val (before, after) = (10, 16)
  var tag2Vec : Map[String, Array[Double]] = Map()

  override def train(dataset: Dataset[_]): AssertionLogRegModel = {
    import dataset.sqlContext.implicits._

    // read the set of all tags
    // val tagSet = inferTagSet(dataset.toDF)
    // assign each tag an array of 3 floats
    // tag2Vec = encode(tagSet)

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.
      withColumn("features", applyWindowUdf(embeddings.get, tag2Vec)($"text", $"pos", $"start", $"end", $"target")).cache()


    /* TODO: pick the parameters you want to expose*/
    val lr = new LogisticRegression()
      .setMaxIter(26)
      .setRegParam(0.001)
      .setElasticNetParam(0.8)


    fillModelEmbeddings(AssertionLogRegModel(lr.fit(processed), tag2Vec))
  }


  def inferTagSet(dataset: Dataset[Row]): Array[String] =
    dataset.select("pos")
      .collect()
      .flatMap { row =>
        row.getAs[Seq[Row]](0).map(_.getString(3)).distinct
      }.distinct


  def encode(tagSet: Array[String]) : Map[String, Array[Double]]= {
    val values = Array(.25, .50, .75, 1)
    val codes = for (a <- values;
                     b <- values;
                     c <- values) yield {
      import math.sqrt
      val norm = sqrt(a * a + b * b + c * c)
      Array(a/norm, b/norm, c/norm)
    }
    tagSet.sorted.zip(codes).toMap
  }
}
