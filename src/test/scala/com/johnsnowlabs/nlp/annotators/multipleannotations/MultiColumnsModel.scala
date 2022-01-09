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

package com.johnsnowlabs.nlp.annotators.multipleannotations

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.util.Identifiable


class MultiColumnsModel(override val uid: String) extends AnnotatorModel[MultiColumnsModel]
  with HasMultipleInputAnnotationCols
  with HasSimpleAnnotate[MultiColumnsModel] {

  def this() = this(Identifiable.randomUID("MERGE"))

  /**
   * Input annotator types: DOCUMEN
   *
   */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT
  /**
   * Output annotator type:DOCUMENT
   *
   */
  override val inputAnnotatorType: AnnotatorType = DOCUMENT


  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations
  }


}


