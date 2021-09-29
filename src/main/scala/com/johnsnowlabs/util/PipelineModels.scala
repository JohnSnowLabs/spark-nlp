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

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.sql.DataFrame


object PipelineModels {

  lazy val dummyDataset: DataFrame = {
    import ResourceHelper.spark.implicits._
    ResourceHelper.spark.createDataset(Seq.empty[String]).toDF("text")
  }

  def apply(stages: Transformer*): PipelineModel = {
    new Pipeline().setStages(stages.toArray).fit(dummyDataset)
  }
}
