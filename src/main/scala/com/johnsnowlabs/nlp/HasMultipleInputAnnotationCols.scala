/*
 * Copyright 2017-2022 John Snow Labs
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

/**
 * Trait  used to create annotators with input columns of variable length.
 * */
trait HasMultipleInputAnnotationCols extends HasInputAnnotationCols {

  /** Annotator reference id. The Annotator type is the same for any of the input columns*/
  val inputAnnotatorType: String

  lazy override val inputAnnotatorTypes: Array[String] = getInputCols.map(_ =>inputAnnotatorType)

  /**
    * Columns that contain annotations necessary to run this annotator
    * AnnotatorType is the same for all input columns in that annotator.
    */
  override def  setInputCols(value: Array[String]): this.type = {
    set(inputCols, value)
  }


}
