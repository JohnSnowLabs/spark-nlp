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

import org.apache.spark.sql.SparkSession

trait HasStorageModel extends HasStorageReader with HasExcludableStorage {

  protected val databases: Array[Database.Name]

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      saveStorage(path, spark, withinStorage = true)
  }

  def saveStorage(path: String, spark: SparkSession, withinStorage: Boolean = false): Unit = {
    databases.foreach(database => {
      StorageHelper.save(path, getReader(database).getConnection, spark, withinStorage)
    })
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    serializeStorage(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      databases.foreach(database => {
        StorageHelper.load(
          path,
          spark,
          database.toString,
          $(storageRef),
          withinStorage = true
        )
      })
  }


   def getEveryIndexInDb(database : Database.Name): List[String] = {
    //Returns a array of String Indexes. These represent every Word coverd by the Embedding Object via RocksDb
    getReader(database).getEveryDbIndex()
  }


}
