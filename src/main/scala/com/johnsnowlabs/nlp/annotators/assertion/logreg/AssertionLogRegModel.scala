package com.johnsnowlabs.nlp.annotators.assertion.logreg

import com.johnsnowlabs.nlp.AnnotatorType.{ASSERTION, DOCUMENT, POS}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType}
import com.johnsnowlabs.nlp.embeddings.{ModelWithWordEmbeddings, WordEmbeddings}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}

/**
  * Created by jose on 22/11/17.
  */

class AssertionLogRegModel(model:LogisticRegressionModel, override val uid: String = Identifiable.randomUID("ASSERTION"))
  extends ModelWithWordEmbeddings[AssertionLogRegModel] with Windowing {
  override val (before, after) = (11, 13)
  override val tokenizer: Tokenizer = new SimpleTokenizer
  override protected def annotate(annotations: Seq[Annotation]): Seq[Annotation] = annotations
  override val annotatorType: AnnotatorType = AnnotatorType.ASSERTION
  override val requiredAnnotatorTypes = Array(DOCUMENT) //, POS
  override final def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sqlContext.implicits._

    /* apply UDF to fix the length of each document */
    val processed = dataset.toDF.
      withColumn("features", applyWindowUdf($"text", $"target", $"start", $"end"))

    super.transform(model.transform(processed))
  }
  override lazy val wordVectors: Option[WordEmbeddings] = embeddings
}

object AssertionLogRegModel {
  def apply(model: LogisticRegressionModel): AssertionLogRegModel = new AssertionLogRegModel(model)
}
