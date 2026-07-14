/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.nlp.serialization

import java.io.{ObjectInputStream, ObjectOutputStream}
import scala.collection.mutable

/** Compatibility reader for mutable.HashMap instances serialized with Scala 2.12.
  *
  * Scala 2.12 mutable.HashMap used serialVersionUID = 1L and a custom HashTable writeObject
  * format: default fields, load factor, size, seed, size-map flag, followed by key/value entry
  * pairs. Scala 2.13 changed the implementation/SUID, so replacing the stream descriptor directly
  * with Scala 2.13 mutable.HashMap causes the stream to be read with the wrong layout and can
  * fail with `StreamCorruptedException: invalid type code: 00`.
  */
@SerialVersionUID(1L)
private[serialization] class LegacyMutableHashMap[A, B]
    extends mutable.AbstractMap[A, B]
    with Serializable {

  @transient
  private var underlying: mutable.HashMap[A, B] = mutable.HashMap.empty[A, B]

  override def get(key: A): Option[B] = underlying.get(key)

  override def iterator: Iterator[(A, B)] = underlying.iterator

  override def addOne(elem: (A, B)): this.type = {
    underlying += elem
    this
  }

  override def subtractOne(key: A): this.type = {
    underlying -= key
    this
  }

  @throws[java.io.IOException]
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    out.writeInt(450) // Scala 2.12 HashTable.defaultLoadFactor
    out.writeInt(underlying.size)
    out.writeInt(0) // seedvalue; irrelevant for compatibility writes
    out.writeBoolean(false) // isSizeMapDefined
    underlying.foreach { case (key, value) =>
      out.writeObject(key)
      out.writeObject(value)
    }
  }

  @throws[java.io.IOException]
  @throws[ClassNotFoundException]
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    underlying = mutable.HashMap.empty[A, B]

    val loadFactor = in.readInt()
    require(loadFactor > 0, s"Invalid legacy mutable.HashMap load factor: $loadFactor")

    val size = in.readInt()
    require(size >= 0, s"Invalid legacy mutable.HashMap size: $size")

    in.readInt() // seedvalue
    in.readBoolean() // isSizeMapDefined

    var index = 0
    while (index < size) {
      val key = in.readObject().asInstanceOf[A]
      val value = in.readObject().asInstanceOf[B]
      underlying += (key -> value)
      index += 1
    }
  }

  private def readResolve(): AnyRef = underlying
}
