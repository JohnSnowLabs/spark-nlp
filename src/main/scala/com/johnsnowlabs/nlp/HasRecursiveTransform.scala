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

import org.apache.spark.ml.{Model, PipelineModel}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, Dataset}

trait HasRecursiveTransform[M <: Model[M]] {

  this: AnnotatorModel[M] =>

  def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation]

  def dfRecAnnotate(recursivePipeline: PipelineModel): UserDefinedFunction = udf {
    annotationProperties: Seq[AnnotationContent] =>
      annotate(annotationProperties.flatMap(_.map(Annotation(_))), recursivePipeline)
  }

  def recursiveTransform(dataset: Dataset[_], recursivePipeline: PipelineModel): DataFrame = {
    _transform(dataset, Some(recursivePipeline))
  }

}
