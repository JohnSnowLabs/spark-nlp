package com.johnsnowlabs.storage

import com.johnsnowlabs.nlp.HasCaseSensitiveProperties

import scala.collection.mutable.{Map => MMap}

trait HasStorageReader[A, B <: StorageReader[A]] extends HasCaseSensitiveProperties {

  @transient protected var readers: MMap[String, B] = _

  protected def createReader(database: Database.Name): B

  protected def getReader(database: Database.Name): B = {
    lazy val reader = createReader(database)
    if (Option(readers).isDefined) {
      readers.getOrElseUpdate(database.toString, reader)
    } else {
      readers = MMap(database.toString -> reader)
      reader
    }
  }

}
