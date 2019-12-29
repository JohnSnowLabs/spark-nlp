package com.johnsnowlabs.storage

import scala.collection.mutable.{Map => MMap}

trait HasStorageReader {

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

}
