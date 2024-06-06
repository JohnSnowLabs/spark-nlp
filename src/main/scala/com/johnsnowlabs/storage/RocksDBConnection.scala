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

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkFiles
import org.rocksdb.{CompressionType, Options, RocksDB}

import java.io.File

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
    options.setMergeOperatorName("stringappend")

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
      require(
        new File(localFromClusterPath).exists(),
        s"Storage not found under given ref: $path\n" +
          "This usually means:\n" +
          "1. You have not loaded any storage under such ref or one of your Storage based " +
          "annotators has `includeStorage` set to false and must be loaded manually\n" +
          "2. You are trying to use cluster mode without a proper shared filesystem.\n" +
          "3. You are trying to use a Kubernetes cluster without a proper shared filesystem. " +
          "In this case, try to enable the parameter to keep models in memory " +
          "(setEnableInMemoryStorage) if available.\n" +
          "4. Your source was not provided to storage creation\n" +
          "5. If you are trying to utilize Storage defined elsewhere, make sure it has the " +
          "appropriate ref. ")
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
    } else if (Option(db).isDefined)
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
    if (cache.contains(pathOrLocator)) cache(pathOrLocator)
    else new RocksDBConnection(pathOrLocator)
  }

  def getOrCreate(database: String, refName: String): RocksDBConnection = {
    val combinedName = StorageHelper.resolveStorageName(database, refName)
    getOrCreate(combinedName)
  }

  def getOrCreate(database: Database.Name, refName: String): RocksDBConnection =
    getOrCreate(database.toString, refName)

  def getLocalPath(fileName: String): String = {
    Path
      .mergePaths(new Path(SparkFiles.getRootDirectory()), new Path("/storage/" + fileName))
      .toString
  }

}
