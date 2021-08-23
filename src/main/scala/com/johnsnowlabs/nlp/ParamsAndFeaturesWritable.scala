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

import org.apache.spark.ml.param.Params
import org.apache.spark.ml.util.{DefaultParamsWritable, MLWriter}
import org.apache.spark.sql.SparkSession

class FeaturesWriter[T](annotatorWithFeatures: HasFeatures, baseWriter: MLWriter, onWritten: (String, SparkSession) => Unit)
  extends MLWriter with HasFeatures {

  override protected def saveImpl(path: String): Unit = {
    baseWriter.save(path)

    for (feature <- annotatorWithFeatures.features) {
      if (feature.orDefault.isDefined)
        feature.serializeInfer(sparkSession, path, feature.name, feature.getOrDefault)
    }

    onWritten(path, sparkSession)

  }
}

trait ParamsAndFeaturesWritable extends DefaultParamsWritable with Params with HasFeatures {

  protected def onWrite(path: String, spark: SparkSession): Unit = {}

  override def write: MLWriter = {
    new FeaturesWriter(
      this,
      super.write,
      (path: String, spark: SparkSession) => onWrite(path, spark)
    )
  }

}
