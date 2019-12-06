package com.johnsnowlabs.storage

import org.rocksdb.{CompressionType, Options, RocksDB}

private[storage] trait RocksDBConnection {

  @transient private var db: RocksDB = _

  def load(): Unit = {
    RocksDB.loadLibrary()
  }

  private def getOptions: Options = {
    val options = new Options()
    options.setCreateIfMissing(true)
    options.setCompressionType(CompressionType.NO_COMPRESSION)
    options.setWriteBufferSize(20 * 1 << 20)
    options
  }

  def open(dbFile: String): RocksDB = {
    load()
    if (Option(db).isDefined)
      db
    else {
      db = RocksDB.open(getOptions, dbFile)
      db
    }
  }

  def openReadOnly(dbFile: String): RocksDB = {
    load()
    if (Option(db).isDefined)
      db
    else {
      db = RocksDB.openReadOnly(getOptions, dbFile)
      db
    }
  }

  def close(): Unit = {
    db.close()
    db = null
  }

  def getDb: RocksDB = {
    if (Option(db).isEmpty)
      throw new Exception("WTF")
    db
  }

  def isOpen: Boolean = {
    if (Option(db).isDefined)
      true
    else
      false
  }

}
