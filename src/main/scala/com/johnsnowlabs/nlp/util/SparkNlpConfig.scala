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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.util.Version
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row}

/** Additional configure options that used by spark.nlp */
object SparkNlpConfig {

  def getEncoder(inputDataset: Dataset[_], newStructType: StructType): ExpressionEncoder[Row] = {
    val sparkVersion = Version.parse(inputDataset.sparkSession.version).toFloat
    if (sparkVersion >= 3.5f) {
      val expressionEncoderClass =
        Class.forName("org.apache.spark.sql.catalyst.encoders.ExpressionEncoder")
      val applyMethod = expressionEncoderClass.getMethod("apply", classOf[StructType])
      applyMethod.invoke(null, newStructType).asInstanceOf[ExpressionEncoder[Row]]
    } else {
      try {
        // Use reflection to access RowEncoder.apply in older Spark versions
        val rowEncoderClass = Class.forName("org.apache.spark.sql.catalyst.encoders.RowEncoder")
        val applyMethod = rowEncoderClass.getMethod("apply", classOf[StructType])
        applyMethod.invoke(null, newStructType).asInstanceOf[ExpressionEncoder[Row]]
      } catch {
        case _: Throwable =>
          throw new UnsupportedOperationException(
            "RowEncoder.apply is not supported in this Spark version.")
      }
    }
  }

}
