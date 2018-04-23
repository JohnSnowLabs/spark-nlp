package com.johnsnowlabs.nlp

import org.apache.spark.ml.param.{Params, StringArrayParam}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

trait HasInputAnnotationCols extends Params {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  val requiredAnnotatorTypes: Array[String]

  /**
    * columns that contain annotations necessary to run this annotator
    * AnnotatorType is used both as input and output columns if not specified
    */
  protected final val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "the input annotation columns")

  /**
    * takes a [[Dataset]] and checks to see if all the required annotation types are present.
    * @param schema to be validated
    * @return True if all the required types are present, else false
    */
  protected def validate(schema: StructType): Boolean = requiredAnnotatorTypes.forall {
    requiredAnnotatorType =>
      schema.exists {
        field => {
          field.metadata.contains("annotatorType") &&
            field.metadata.getString("annotatorType") == requiredAnnotatorType &&
            $(inputCols).contains(field.name)
        }
      }
  }

  /** Overrides required annotators column if different than default */
  final def setInputCols(value: Array[String]): this.type = {
    require(
      value.length == requiredAnnotatorTypes.length,
      s"setInputCols expecting ${requiredAnnotatorTypes.length} columns. " +
        s"Provided column amount: ${value.length} " +
        s"which should be made of: ${requiredAnnotatorTypes.mkString(", ")} annotators"
    )
    set(inputCols, value)
  }

  final def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** @return input annotations columns currently used */
  final def getInputCols: Array[String] =
    get(inputCols).orElse(getDefault(inputCols))
      .getOrElse(throw new Exception(s"inputCols not provided." +
      s" Requires columns for ${requiredAnnotatorTypes.mkString(", ")} annotators"))
}
