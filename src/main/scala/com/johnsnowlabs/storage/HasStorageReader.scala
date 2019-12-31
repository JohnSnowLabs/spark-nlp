package com.johnsnowlabs.storage

import scala.collection.mutable.{Map => MMap}

trait HasStorageReader {

  @transient protected var readers: MMap[String, StorageReader[_]] = _

  protected def createReader(database: Database.Name): StorageReader[_]

  protected def getReader[A](database: Database.Name): StorageReader[A] = {
    lazy val reader = createReader(database)
    if (Option(readers).isDefined) {
      readers.getOrElseUpdate(database.toString, reader).asInstanceOf[StorageReader[A]]
    } else {
      readers = MMap(database.toString -> reader)
      reader.asInstanceOf[StorageReader[A]]
    }
  }

}
