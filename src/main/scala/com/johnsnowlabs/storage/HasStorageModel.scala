package com.johnsnowlabs.storage

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.SparkSession
import scala.collection.mutable.{Map => MMap}

trait HasStorageModel extends HasStorageProperties {

  @transient protected var readers: MMap[String, StorageReader[_]] = _

  protected def createReader(database: String): StorageReader[_]

  protected def getReader[A](database: String): StorageReader[A] = {
    lazy val reader = createReader(database)
    if (Option(readers).isDefined) {
      readers.getOrElseUpdate(database, reader).asInstanceOf[StorageReader[A]]
    } else {
      readers = MMap(database -> reader)
      reader.asInstanceOf[StorageReader[A]]
    }
  }

  def serializeStorage(path: String, spark: SparkSession): Unit = {
    databases.foreach(database => {
      val index = new Path(RocksDBConnection.getLocalPath(getReader(database).getConnection.getFileName))

      val uri = new java.net.URI(path)
      val fs = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
      val dst = getStorageSerializedPath(path)

      StorageHelper.save(fs, index, dst)
    })
  }

  override protected def onWrite(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage))
      serializeStorage(path, spark)
  }

  def deserializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(includeStorage)) {
      val src = getStorageSerializedPath(path)
      databases.foreach(database =>
      StorageHelper.load(
        src.toUri.toString,
        spark,
        database
      ))
    }
  }

}
