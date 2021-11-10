/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.CHUNK
import org.apache.spark.ml.param.{Params, StringArrayParam}
import org.apache.spark.sql.types.StructType

trait HasMultipleInputAnnotationCols extends HasInputAnnotationCols {

  val inputAnnotatorType: String

  lazy override val inputAnnotatorTypes: Array[String] = getInputCols.map(_ =>inputAnnotatorType)

  override def  setInputCols(value: Array[String]): this.type = {
    set(inputCols, value)
  }


}
