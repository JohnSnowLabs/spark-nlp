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

package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.Param
import org.apache.spark.sql.Dataset

trait HasStorageRef extends ParamsAndFeaturesWritable {

  /** Unique identifier for storage (Default: `this.uid`)
    * WARNING: this is for internal use and not intended for users
    * @group param
    */
  val storageRef = new Param[String](this, "storageRef", "storage unique identifier")

  setDefault(storageRef, this.uid)

  def createDatabaseConnection(database: Database.Name): RocksDBConnection =
    RocksDBConnection.getOrCreate(database, $(storageRef))

  def setStorageRef(value: String): this.type = {
    if (get(storageRef).nonEmpty)
      throw new UnsupportedOperationException(
        s"Cannot override storage ref on $this. " +
          s"Please re-use current ref: $getStorageRef")
    set(this.storageRef, value)
  }
  def getStorageRef: String = $(storageRef)

  def validateStorageRef(
      dataset: Dataset[_],
      inputCols: Array[String],
      annotatorType: String): Unit = {
    require(
      isDefined(storageRef),
      "This Annotator does not have a storage reference defined. This could be an outdated " +
        "model or an incorrectly created one. Make sure storageRef param is defined and set.")
    require(
      HasStorageRef.getStorageRefFromInput(dataset, inputCols, annotatorType) == $(storageRef),
      s"Found input column with storage metadata. But such ref does not match to the ref this annotator requires. " +
        s"Make sure you are loading the annotator with ref: ${$(storageRef)}")
  }

}

object HasStorageRef {
  def getStorageRefFromInput(
      dataset: Dataset[_],
      inputCols: Array[String],
      annotatorType: String): String = {
    val storageCol = dataset.schema.fields
      .find(f =>
        inputCols.contains(f.name) && f.metadata.getString("annotatorType") == annotatorType)
      .getOrElse(throw new Exception(
        s"Could not find a column of type $annotatorType. Make sure your pipeline is correct."))
      .name

    val storage_meta = dataset.select(storageCol).schema.fields.head.metadata

    require(
      storage_meta.contains("ref"),
      s"Could not find a ref name in column $storageCol. " +
        s"Make sure $storageCol was created appropriately with a valid storageRef")

    storage_meta.getString("ref")
  }
}
