package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{Params, StringArrayParam}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

trait HasInputAnnotationCols extends Params {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  val inputAnnotatorTypes: Array[String]

  /**
    * columns that contain annotations necessary to run this annotator
    * AnnotatorType is used both as input and output columns if not specified
    */
  protected final val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "the input annotation columns")

  /** Overrides required annotators column if different than default */
  final def setInputCols(value: Array[String]): this.type = {
    require(
      value.length == inputAnnotatorTypes.length,
      s"setInputCols in ${this.uid} expecting ${inputAnnotatorTypes.length} columns. " +
        s"Provided column amount: ${value.length}. " +
        s"Which should be columns from the following annotators: ${inputAnnotatorTypes.mkString(", ")}"
    )
    set(inputCols, value)
  }

  protected def checkSchema(schema: StructType, inputAnnotatorType: String): Boolean = {
    schema.exists {
      field => {
        field.metadata.contains("annotatorType") &&
          field.metadata.getString("annotatorType") == inputAnnotatorType &&
          getInputCols.contains(field.name)
      }
    }
  }

  final def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** @return input annotations columns currently used */
  def getInputCols: Array[String] =
    get(inputCols).orElse(getDefault(inputCols))
      .getOrElse(throw new Exception(s"inputCols not provided." +
      s" Requires columns for ${inputAnnotatorTypes.mkString(", ")} annotators"))
}
