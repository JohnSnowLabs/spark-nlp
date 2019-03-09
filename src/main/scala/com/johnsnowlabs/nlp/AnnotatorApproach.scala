package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{ParamMap, StringArrayParam}
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
    with HasOutputAnnotatorType
    with DefaultParamsWritable {

  val description: String

  def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel] = None): M

  def beforeTraining(spark: SparkSession): Unit = {}

  def onTrained(model: M, spark: SparkSession): Unit = {}

  val trainingAnnotatorTypes: Array[String] = inputAnnotatorTypes

  /**
    * columns that contain annotations necessary to train this annotator
    * AnnotatorType is used in the same way than input and output annotator types
    */
  protected final val trainingCols: StringArrayParam =
    new StringArrayParam(this, "trainingCols", "the training annotation columns. uses input annotation columns if missing")

  /** Overrides required annotators column if different than default */
  final def setTrainingCols(value: Array[String]): this.type = {
    set(trainingCols, value)
  }

  final def setTrainingCols(value: String*): this.type = {
    require(
      value.length == trainingAnnotatorTypes.length,
      s"setInputCols in ${this.uid} expecting ${inputAnnotatorTypes.length} columns. " +
        s"Provided column amount: ${value.length}. " +
        s"Which should be columns from the following annotators: ${inputAnnotatorTypes.mkString(", ")}"
    )
    set(trainingCols, value.toArray)
  }

  final def getTrainingCols: Array[String] = $(trainingCols)

  override def getInputCols: Array[String] = {
    get(trainingCols).getOrElse(super.getInputCols)
  }

  /**
    * takes a [[Dataset]] and checks to see if all the required annotation types are present.
    * @param schema to be validated
    * @return True if all the required types are present, else false
    */
  protected def validate(schema: StructType): Boolean = {
    getTrainingCols.forall {
      inputAnnotatorType =>
        checkSchema(schema, inputAnnotatorType)
    }
  }

  override final def fit(dataset: Dataset[_]): M = {
    beforeTraining(dataset.sparkSession)
    val model = copyValues(train(dataset).setParent(this))
    onTrained(model, dataset.sparkSession)
    model
  }

  override final def copy(extra: ParamMap): Estimator[M] = defaultCopy(extra)

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    require(validate(schema), s"Wrong or missing inputCols annotators in $uid. " +
      s"Received inputCols: ${$(inputCols).mkString(",")}. Make sure such columns have following annotator types: " +
      s"${inputAnnotatorTypes.mkString(", ")}")
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }
}
