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

package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.{HasFeatures, ParamsAndFeaturesReadable}
import org.apache.spark.sql.SparkSession

trait StorageReadable[T <: HasStorageModel with HasFeatures] extends ParamsAndFeaturesReadable[T] {

  val databases: Array[Database.Name]

  def loadStorage(path: String, spark: SparkSession, storageRef: String): Unit = {
    databases.foreach(database => {
      StorageHelper.load(
        path,
        spark,
        database.toString,
        storageRef,
        withinStorage = false
      )
    })
  }

  def readStorage(instance: T, path: String, spark: SparkSession): Unit = {
    instance.deserializeStorage(path, spark)
  }

  addReader(readStorage)
}
