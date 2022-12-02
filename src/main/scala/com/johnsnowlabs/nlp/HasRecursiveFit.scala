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

import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.Dataset

/** AnnotatorApproach'es may extend this trait in order to allow RecursivePipelines to include
  * intermediate steps trained PipelineModel's
  */
trait HasRecursiveFit[M <: Model[M]] {

  this: AnnotatorApproach[M] =>

  final def recursiveFit(dataset: Dataset[_], recursivePipeline: PipelineModel): M = {
    _fit(dataset, Some(recursivePipeline))
  }

}
