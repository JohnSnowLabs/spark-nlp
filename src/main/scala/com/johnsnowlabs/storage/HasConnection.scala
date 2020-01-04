package com.johnsnowlabs.storage

trait HasConnection extends AutoCloseable {

  protected val connection: RocksDBConnection

  this match {
    case _: StorageBatchWriter[_] => connection.connectReadWrite
    case _: StorageReadWriter[_] => connection.connectReadWrite
    case _ => connection.connectReadOnly
  }

  override def close(): Unit = {
    connection.close()
  }

  def getConnection: RocksDBConnection = connection

}
