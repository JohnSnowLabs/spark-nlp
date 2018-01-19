package com.johnsnowlabs.nlp.annotators.assertion.svm

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.assertion.logreg._
import com.johnsnowlabs.nlp.annotators.assertion.logreg.mllib.vectors
import com.johnsnowlabs.nlp.embeddings.{AnnotatorWithWordEmbeddings, WordEmbeddings}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row}
import smile.classification.SVM
import smile.classification.SVM.Multiclass
import smile.math.kernel.GaussianKernel

/**
  * Created by jose on 18/12/17.
  */
class AssertionSVMApproach(override val uid: String) extends
  AnnotatorWithWordEmbeddings[AssertionSVMApproach, AssertionSVMModel] with Windowing {

  override val requiredAnnotatorTypes = Array(DOCUMENT, POS)
  override val description: String = "Clinical Text Status Assertion"
  override val tokenizer: Tokenizer = new SimpleTokenizer
  override lazy val wordVectors: Option[WordEmbeddings] = embeddings
  val beforeParam = new Param[Int](this, "before", "Length of the context before the target")
  val afterParam = new Param[Int](this, "after", "Length of the context after the target")
  // example of possible values, 'Negated', 'Affirmed', 'Historical'
  val labelColumn = new Param[String](this, "label", "Column with one label per document")


  val gammaParam = new Param[Double](this, "gamma", "The gamma for the kernel")
  val CParam = new Param[Double](this, "C", "The C for the margin")
  def setLabelCol(label: String) = set(labelColumn, label)
  def setBefore(before: Int) = set(beforeParam, before)
  def setAfter(after: Int) = set(afterParam, after)

  def setGamma(gamma: Double) = set(gammaParam, gamma)
  def setC(c: Double) = set(CParam, c)

  var tag2Vec : Map[String, Array[Double]] = Map()

  def inferTagSet(dataset: Dataset[Row]): Array[String] =
    dataset.select("pos")
      .collect()
      .flatMap { row =>
        row.getAs[Seq[Row]](0).map(_.getString(3)).distinct
      }.distinct

  def encode(tagSet: Array[String]) : Map[String, Array[Double]]= {
    val values = Array(.25, .50, .75, 1.0)
    val codes = for (a <- values;
                     b <- values;
                     c <- values) yield {
      import math.sqrt
      val norm = sqrt(a * a + b * b + c * c)
      Array(a/norm, b/norm, c/norm)
    }
    tagSet.sorted.zip(codes).toMap
  }

  override val annotatorType: AnnotatorType = ASSERTION
  def this() = this(Identifiable.randomUID("ASSERTION"))

  override def train(dataset: Dataset[_]): AssertionSVMModel = {
    import dataset.sqlContext.implicits._

    val tagSet = inferTagSet(dataset.toDF)
    val tagMap = encode(tagSet)
    /* apply UDF to fix the length of each document */
    val theUdf : UserDefinedFunction= applyWindowAndPOSUdf(vectors, tagMap)
    val processed = dataset.toDF.
      withColumn("features", theUdf($"text", $"target", $"start", $"end", $"pos"))

    val labelCol = getOrDefault(labelColumn)

    /* infer labels and assign a number to each
    val labelMappings: Map[String, Double] = dataset.select(labelCol).distinct.collect
      .map(row => row.getAs[String](labelCol)).zipWithIndex
      .map{case (label, idx) => (label, idx.toDouble)}
      .toMap
      */

    val labelMappings = Map("hypothetical" -> 0.0,
      "present" -> 1.0, "absent" -> 2.0, "possible" -> 3.0,
      "conditional"-> 4.0, "associated_with_someone_else" -> 5.0)

    val processedWithLabel = processed.withColumn(labelCol, labelToNumber(labelMappings)(col(labelCol)))


    /* collect and separate feature and labels */
    val features = processedWithLabel.select($"features").collect.map(_.getAs[Vector](0).toArray)
    val labels = processedWithLabel.select($"label").collect.map(_.getAs[Double](0))

    //7, 80 -> 90.36
    //6, 80 -> 90.42
    //5, 80 -> 90.52
    //5, 85 -> 90,68
    //5, 75 -> 90.57
    //1, 80 -> 78.12


    val svm = new SVM(new GaussianKernel(5.0), 85.0, 6, Multiclass.ONE_VS_ONE)
    svm.learn(features, labels.map(_.toInt))
    svm.learn(features, labels.map(_.toInt))
    svm.learn(features, labels.map(_.toInt))
    svm.learn(features, labels.map(_.toInt))
    svm.learn(features, labels.map(_.toInt))
    svm.finish()

    AssertionSVMModel()
      .setModel(svm)
      .setBefore(getOrDefault(beforeParam))
      .setAfter(getOrDefault(afterParam))
      .setLabelMap(labelMappings)
      .setTagMap(tagMap)

  }

  private def labelToNumber(mappings: Map[String, Double]) = udf { label:String  => mappings.get(label)}

  override val before: Int = 11
  override val after: Int = 13
}
