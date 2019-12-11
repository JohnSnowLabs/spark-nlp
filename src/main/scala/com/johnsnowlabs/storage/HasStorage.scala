package com.johnsnowlabs.storage

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{BooleanParam, Param, Params}
import org.apache.spark.sql.{Dataset, SparkSession}

trait HasStorage[A] extends Params {

  val includeStorage = new BooleanParam(this, "includeStorage", "whether or not to save indexed storage along this annotator")
  val storageRef = new Param[String](this, "storageRef", "unique reference name for identification")
  val storagePath = new Param[String](this, "storagePath", "path to file")
  val storageFormat = new Param[String](this, "storageFormat", "file format")

  protected def loader: StorageLoader

  def setStoragePath(path: String): this.type = set(storagePath, path)

  def getStoragePath: String = $(storagePath)

  def setStorageFormat(format: String): this.type = {
    set(storageFormat, format)
  }

  def getStorageFormat: String = {
    $(storageFormat)
  }

  setDefault(includeStorage, true)

  def loadStorage(spark: SparkSession): Unit = {
    require(isDefined(storagePath) || isDefined(storageRef),
      s"Word embeddings not found. Either sourceEmbeddingsPath not set," +
        s" or not in cache by ref: ${get(storageRef).getOrElse("-storageRef not set-")}. " +
        s"Load using EmbeddingsHelper.load() and .setStorageRef() to make them available."
    )
    if (isDefined(storagePath)) {
      loader.load(
        $(storagePath),
        spark,
        $(storageFormat),
        $(storageRef)
      )
    }
  }

  def setIncludeStorage(value: Boolean): this.type = set(includeStorage, value)
  def getIncludeStorage: Boolean = $(includeStorage)

  def setStorageRef(value: String): this.type = {
    if (get(storageRef).nonEmpty)
      throw new UnsupportedOperationException(s"Cannot override storage ref on $this. " +
        s"Please re-use current ref: $getStorageRef")
    set(this.storageRef, value)
  }
  def getStorageRef: String = $(storageRef)

  def validateStorageRef(dataset: Dataset[_], inputCols: Array[String], annotatorType: String): Unit = {
    val storageCol = dataset.schema.fields
      .find(f => inputCols.contains(f.name) && f.metadata.getString("annotatorType") == annotatorType)
      .getOrElse(throw new Exception(s"Could not find a valid storage column, make sure the storage is loaded " +
        s"and has the following ref: ${$(storageRef)}")).name

    val storage_meta = dataset.select(storageCol).schema.fields.head.metadata

    require(storage_meta.contains("ref"), "Cannot find storage ref in column schema metadata")

    require(storage_meta.getString("ref") == $(storageRef),
      s"Found storage column, but ref does not match to the ref this model was trained with. " +
        s"Make sure you are using the right storage in your pipeline, with ref: ${$(storageRef)}")
  }

  protected def getStorageSerializedPath(path: String): Path =
    Path.mergePaths(new Path(path), new Path("/storage"))

}
