package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model
import org.apache.spark.sql.types._

/**
  * Created by jose on 25/01/18.
  */
trait TransformModelSchema {

  this: Model[_] with HasOutputAnnotationCol with HasAnnotatorType =>

  /** Shape of annotations at output */
  private def outputDataType: DataType = ArrayType(Annotation.dataType)

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, outputDataType, nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }

}
