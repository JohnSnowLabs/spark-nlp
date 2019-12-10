package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.ParamsAndFeaturesWritable
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession

trait HasStorageModel[A] extends HasStorage[A] with ParamsAndFeaturesWritable {

  protected val reader: RocksDBReader[A]

  def serializeEmbeddings(path: String, spark: SparkSession): Unit = {
    val index = new Path(RocksDBConnection.getLocalPath(getStorageConnection.getFileName))

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

      if (!storageIsReady) {
        loader.load(
          src.toUri.toString,
          spark,
          loader.formats.SPARKNLP,
          $(storageRef)
        )
        setStorageConnection()
      }
    }
  }

}
