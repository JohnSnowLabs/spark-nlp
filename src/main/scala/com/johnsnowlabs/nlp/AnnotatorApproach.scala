package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.{Estimator, Model, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}
import org.apache.spark.ml.util.DefaultParamsWritable

/**
  * Created by saif on 12/06/2017.
  */

/** This class should grow once we start training on datasets and share params
  * For now it stands as a dummy placeholder for future reference
  */
abstract class AnnotatorApproach[M <: Model[M]]
  extends Estimator[M]
    with HasInputAnnotationCols
    with HasOutputAnnotationCol
    with HasAnnotatorType
    with DefaultParamsWritable {

  val description: String

  def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel] = None): M

  def beforeTraining(spark: SparkSession): Unit = {}

  def onTrained(model: M, spark: SparkSession): Unit = {}

  override final def fit(dataset: Dataset[_]): M = {
    beforeTraining(dataset.sparkSession)
    val model = copyValues(train(dataset).setParent(this))
    onTrained(model, dataset.sparkSession)
    model
  }

  override final def copy(extra: ParamMap): Estimator[M] = defaultCopy(extra)

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    require(validate(schema), s"Missing annotators in pipeline. Make sure the following are present: " +
      s"${requiredAnnotatorTypes.mkString(", ")}")
    getInputCols.foreach {
      annotationColumn =>
        require(getInputCols.forall(schema.fieldNames.contains),
          s"pipeline annotator stages incomplete. " +
            s"expected: ${getInputCols.mkString(", ")}, " +
            s"found: ${schema.fields.filter(_.dataType == ArrayType(Annotation.dataType)).map(_.name).mkString(", ")}, " +
            s"among available: ${schema.fieldNames.mkString(", ")}")
        require(schema(annotationColumn).dataType == ArrayType(Annotation.dataType),
          s"column [$annotationColumn] must be an NLP Annotation column")
    }
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }
}
