/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import org.apache.spark.ml.Model
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Column, Dataset}
import org.apache.spark.sql.types._


trait RawAnnotator[M <: Model[M]] extends Model[M]
  with ParamsAndFeaturesWritable
  with HasOutputAnnotatorType
  with HasInputAnnotationCols
  with HasOutputAnnotationCol {

  /** Shape of annotations at output */
  private def outputDataType: DataType = ArrayType(Annotation.dataType)

  protected def wrapColumnMetadata(col: Column): Column = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    col.as(getOutputCol, metadataBuilder.build)
  }

  /**
   * takes a [[Dataset]] and checks to see if all the required annotation types are present.
   *
   * @param schema to be validated
   * @return True if all the required types are present, else false
   */
  protected def validate(schema: StructType): Boolean = {
    inputAnnotatorTypes.forall {
      inputAnnotatorType =>
        checkSchema(schema, inputAnnotatorType)
    }
  }

  /** Override for additional custom schema checks */
  protected def extraValidateMsg = "Schema validation failed"

  protected def extraValidate(structType: StructType): Boolean = {
    true
  }

  /** requirement for pipeline transformation validation. It is called on fit() */
  override final def transformSchema(schema: StructType): StructType = {
    require(extraValidate(schema), extraValidateMsg)
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, outputDataType, nullable = false, metadataBuilder.build)
    StructType(outputFields)
  }


  /** requirement for annotators copies */
  override def copy(extra: ParamMap): M = defaultCopy(extra)
}
