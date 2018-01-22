package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.MetadataBuilder

/**
  * Created by jose on 21/01/18.
  */
abstract class AnnotatorModel[M <: Model[M]] extends BaseAnnotatorModel[M] {

  /**
    * Given requirements are met, this applies ML transformation within a Pipeline or stand-alone
    * Output annotation will be generated as a new column, previous annotations are still available separately
    * metadata is built at schema level to record annotations structural information outside its content
    *
    * @param dataset [[Dataset[Row]]]
    * @return
    */
  override final def transform(dataset: Dataset[_]): DataFrame = {
    require(validate(dataset.schema), s"Missing annotators in pipeline. Make sure the following are present: " +
      s"${requiredAnnotatorTypes.mkString(", ")}")
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    dataset.withColumn(
      getOutputCol,
      dfAnnotate(
        array(getInputCols.map(c => dataset.col(c)):_*)
      ).as(getOutputCol, metadataBuilder.build)
    )
  }

  /**
    * Wraps annotate to happen inside SparkSQL user defined functions in order to act with [[org.apache.spark.sql.Column]]
    * @return udf function to be applied to [[inputCols]] using this annotator's annotate function as part of ML transformation
    */
  private def dfAnnotate: UserDefinedFunction = udf {
    annotatorProperties: Seq[AnnotationContent] =>
      annotate(annotatorProperties.flatMap(_.map(Annotation(_))))
  }

  /**
    * internal types to show Rows as a relevant StructType
    * Should be deleted once Spark releases UserDefinedTypes to @developerAPI
    */
  private type AnnotationContent = Seq[Row]

}
