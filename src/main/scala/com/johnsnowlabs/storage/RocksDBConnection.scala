package com.johnsnowlabs.storage

import java.io.File

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkFiles
import org.rocksdb.{CompressionType, Options, RocksDB}

final class RocksDBConnection private (path: String) extends AutoCloseable {

  RocksDB.loadLibrary()
  @transient private var db: RocksDB = _

  def getFileName: String = path

  private def getOptions: Options = {
    val options = new Options()
    options.setCreateIfMissing(true)
    options.setCompressionType(CompressionType.NO_COMPRESSION)
    options.setWriteBufferSize(20 * 1 << 20)
    options.setKeepLogFileNum(1)
    options.setDbLogDir(System.getProperty("java.io.tmpdir"))

    options
  }

  def findLocalIndex: String = {
    val localPath = RocksDBConnection.getLocalPath(path)
    if (new File(localPath).exists()) {
      localPath
    } else if (new File(path).exists()) {
      path
    } else {
      val localFromClusterPath = SparkFiles.get(path)
      require(new File(localFromClusterPath).exists(), s"Storage not found under given ref: $path\n" +
        s" This usually means:\n1. You have not loaded any storage under such ref or one of your Storage based annotators " +
        s"has `includeStorage` set to false and must be loaded manually\n2." +
        s" You are trying to use cluster mode without a proper shared filesystem.\n3. source was not provided to Storage creation" +
        s"\n4. If you are trying to utilize Storage defined elsewhere, make sure it has the appropriate ref. ")
      localFromClusterPath
    }
  }

  def connectReadWrite: RocksDB = {
    if (Option(db).isDefined) {
      db
    } else {
      db = RocksDB.open(getOptions, findLocalIndex)
      RocksDBConnection.cache.update(path, this)
      db
    }
  }

  def connectReadOnly: RocksDB = {
    if (RocksDBConnection.cache.contains(path)) {
      db = RocksDBConnection.cache(path).getDb
      db
    }
    else if (Option(db).isDefined)
      db
    else {
      db = RocksDB.openReadOnly(getOptions, findLocalIndex)
      RocksDBConnection.cache.update(path, this)
      db
    }
  }

  def reconnectReadWrite: RocksDB = {
    require(Option(db).isDefined, "Tried to reconnect on a closed connection")
    close()
    connectReadWrite
  }

  override def close(): Unit = {
    if (Option(db).isDefined) {
      db.close()
      db = null
      RocksDBConnection.cache.remove(path)
    }
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
  @transient private[storage] val cache: scala.collection.mutable.Map[String, RocksDBConnection] =
    scala.collection.mutable.Map.empty[String, RocksDBConnection]

  def getOrCreate(pathOrLocator: String): RocksDBConnection = {
    if (cache.contains(pathOrLocator)) cache(pathOrLocator) else new RocksDBConnection(pathOrLocator)
  }

  def getOrCreate(database: String, refName: String): RocksDBConnection = {
    val combinedName = StorageHelper.resolveStorageName(database, refName)
    getOrCreate(combinedName)
  }

  def getOrCreate(database: Database.Name, refName: String): RocksDBConnection = getOrCreate(database.toString, refName)

  def getLocalPath(fileName: String): String = {
    Path.mergePaths(new Path(SparkFiles.getRootDirectory()), new Path("/storage/"+fileName)).toString
  }

}