package com.johnsnowlabs.storage

import java.io.File

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkFiles
import org.rocksdb.{CompressionType, Options, RocksDB}

final class RocksDBConnection(dbFile: String) {

  RocksDB.loadLibrary()
  @transient private var db: RocksDB = _

  def getFileName: String = dbFile

  private def getOptions: Options = {
    val options = new Options()
    options.setCreateIfMissing(true)
    options.setCompressionType(CompressionType.NO_COMPRESSION)
    options.setWriteBufferSize(20 * 1 << 20)
    options
  }

  private def findLocalDb: String = {
    lazy val localPath = RocksDBConnection.getLocalPath(dbFile)
    if (new File(dbFile).exists())
      dbFile
    else if (new File(localPath).exists()) {
      localPath
    }
    else {
      val localFromClusterPath = SparkFiles.get(dbFile)
      require(new File(localFromClusterPath).exists(), s"Storage not found under given ref: $dbFile\n" +
        s" This usually means:\n1. You have not loaded any storage under such ref\n2." +
        s" You are trying to use cluster mode without a proper shared filesystem.\n3. source was not provided to Storage creation" +
        s"\n4. If you are trying to utilize Storage defined elsewhere, make sure you it's appropriate ref. ")
      localFromClusterPath
    }
  }

  def connectReadWrite: RocksDB = {
    if (Option(db).isDefined)
      db
    else {
      db = RocksDB.open(getOptions, findLocalDb)
      db
    }
  }

  def connectReadOnly: RocksDB = {
    if (Option(db).isDefined)
      db
    else {
      db = RocksDB.openReadOnly(getOptions, findLocalDb)
      db
    }
  }

  def close(): Unit = {
    db.close()
    db = null
  }

  def getDb: RocksDB = {
    if (Option(db).isEmpty)
      throw new Exception("Attempted to get a non-existing connection")
    db
  }

  def isConnected: Boolean = {
    if (Option(db).isDefined)
      true
    else
      false
  }

}

object RocksDBConnection {
  def getLocalPath(fileName: String): String = Path.mergePaths(new Path(SparkFiles.getRootDirectory()), new Path("/"+fileName)).toString
}