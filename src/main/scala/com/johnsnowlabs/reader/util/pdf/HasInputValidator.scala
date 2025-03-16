/*
 * Copyright 2017-2025 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.reader.util.pdf

import org.apache.spark.sql.types.{ArrayType, DataType, MapType, StructType}

trait HasInputValidator {
  val uid: String

  def compareDataTypes(dtype1: DataType, dtype2: DataType): Boolean = {
    if (dtype1.getClass != dtype2.getClass) {
      return false
    }

    (dtype1, dtype2) match {
      case (a1: ArrayType, a2: ArrayType) =>
        compareDataTypes(a1.elementType, a2.elementType)

      case (s1: StructType, s2: StructType) =>
        if (s1.fields.length != s2.fields.length) {
          return false
        }
        s1.fields.zip(s2.fields).forall { case (field1, field2) =>
          field1.name == field2.name && compareDataTypes(field1.dataType, field2.dataType)
        }

      case (m1: MapType, m2: MapType) =>
        compareDataTypes(m1.keyType, m2.keyType) && compareDataTypes(m1.valueType, m2.valueType)

      case _ =>
        dtype1 == dtype2
    }
  }

  def validateInputCol(schema: StructType, colName: String, colType: DataType) {
    require(
      schema.exists(_.name == colName),
      s"Missing input column in $uid: Column '${colName}' is not present." +
        s"Make sure such transformer exist in your pipeline, " +
        s"with the right output names.")
    require(
      compareDataTypes(schema.find(_.name == colName).map(_.dataType).get, colType),
      s"Column '${colName}' is not a valid ${colType} in $uid")
  }

}
