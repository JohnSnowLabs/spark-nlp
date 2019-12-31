package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.sql.Dataset

trait HasStorageRef extends ParamsAndFeaturesWritable {

  protected val databases: Array[String]

  val storageRef = new Param[String](this, "storageRef", "storage unique identifier")

  setDefault(storageRef, this.uid)

  def setStorageRef(value: String): this.type = {
    if (get(storageRef).nonEmpty)
      throw new UnsupportedOperationException(s"Cannot override storage ref on $this. " +
        s"Please re-use current ref: $getStorageRef")
    set(this.storageRef, value)
  }
  def getStorageRef: String = $(storageRef)

  def getStorageRefFromInput(dataset: Dataset[_], inputCols: Array[String], annotatorType: String): String = {
    val storageCol = dataset.schema.fields
      .find(f => inputCols.contains(f.name) && f.metadata.getString("annotatorType") == annotatorType)
      .getOrElse(throw new Exception(s"Could not find a valid storage column, make sure the storage is loaded " +
        s"and has the following ref: ${$(storageRef)}")).name

    val storage_meta = dataset.select(storageCol).schema.fields.head.metadata

    require(storage_meta.contains("ref"), "Cannot find storage ref in column schema metadata")

    storage_meta.getString("ref")
  }

  def validateStorageRef(dataset: Dataset[_], inputCols: Array[String], annotatorType: String): Unit = {
    require(isDefined(storageRef), "This model does not have a storage reference defined. This could be an outdated model or incorrectly created one. Make sure storageRef param is defined.")
    require(getStorageRefFromInput(dataset, inputCols, annotatorType) == $(storageRef),
      s"Found storage column, but ref does not match to the ref this model was trained with. " +
        s"Make sure you are using the right storage in your pipeline, with ref: ${$(storageRef)}")
  }

}
