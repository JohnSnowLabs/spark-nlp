package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.HasCaseSensitiveProperties

import scala.collection.mutable.{Map => MMap}

trait HasStorageReader extends HasStorageRef with HasCaseSensitiveProperties {

  @transient protected var readers: MMap[Database.Name, StorageReader[_]] = _

  protected def createReader(database: Database.Name, connection: RocksDBConnection): StorageReader[_]

  protected def getReader[A](database: Database.Name): StorageReader[A] = {
    lazy val reader = createReader(database, createDatabaseConnection(database))
    if (Option(readers).isDefined) {
      readers.getOrElseUpdate(database, reader).asInstanceOf[StorageReader[A]]
    } else {
      readers = MMap(database -> reader)
      reader.asInstanceOf[StorageReader[A]]
    }
  }

}
