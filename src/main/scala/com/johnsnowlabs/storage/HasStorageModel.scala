package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

trait HasStorageModel[A] extends HasStorage[A] with ParamsAndFeaturesWritable {

  @transient protected var reader: StorageReader[A] = _

  protected def createReader: StorageReader[A]

  protected def getReader: StorageReader[A] = {
    if (Option(reader).isEmpty) {
      reader = createReader
      reader
    } else
      reader
  }

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val index = new Path(RocksDBConnection.getLocalPath(getReader.getConnection.getFileName))

    val uri = new java.net.URI(path)
    val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val dst = getStorageSerializedPath(path)

    StorageHelper.save(fs, index, dst)
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      serializeEmbeddings(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage)) {
      val src = getStorageSerializedPath(path)
      loader.load(
        src.toUri.toString,
        spark,
        loader.formats.SPARKNLP.toString,
        $(storageRef)
      )
    }
  }

}
