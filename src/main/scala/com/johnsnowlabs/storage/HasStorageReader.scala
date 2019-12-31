package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.HasCaseSensitiveProperties

import scala.collection.mutable.{Map => MMap}

trait HasStorageReader[StorageType, Reader <: StorageReader[StorageType]] extends HasCaseSensitiveProperties {

  @transient protected var readers: MMap[String, Reader] = _

  protected def createReader(database: Database.Name): Reader

  protected def getReader(database: Database.Name): Reader = {
    lazy val reader = createReader(database)
    if (Option(readers).isDefined) {
      readers.getOrElseUpdate(database.toString, reader)
    } else {
      readers = MMap(database.toString -> reader)
      reader
    }
  }

}
