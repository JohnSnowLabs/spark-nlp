/*
 * Copyright 2017-2021 John Snow Labs
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
