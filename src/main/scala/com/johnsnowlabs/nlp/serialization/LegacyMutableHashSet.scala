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

/** Compatibility reader for mutable.HashSet instances serialized with Scala 2.12.
  *
  * Scala 2.12 mutable.HashSet used serialVersionUID = 1L and a custom writeObject format from
  * FlatHashTable: default fields, load factor, size, seed, size-map flag, followed by elements.
  * Scala 2.13 changed the implementation/SUID and made mutable.HashSet final, so replacing the
  * stream descriptor with the 2.13 mutable.HashSet descriptor corrupts the stream. This class
  * consumes the 2.12 wire format and materializes a mutable.Set-compatible instance.
  */
@SerialVersionUID(1L)
private[serialization] class LegacyMutableHashSet[A]
    extends mutable.AbstractSet[A]
    with mutable.Set[A]
    with Serializable {

  @transient
  private var underlying: mutable.Set[A] = mutable.Set.empty[A]

  override def contains(elem: A): Boolean = underlying.contains(elem)

  override def iterator: Iterator[A] = underlying.iterator

  override def addOne(elem: A): this.type = {
    underlying += elem
    this
  }

  override def subtractOne(elem: A): this.type = {
    underlying -= elem
    this
  }

  override def clear(): Unit = underlying.clear()

  @throws[java.io.IOException]
  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    out.writeInt(450) // Scala 2.12 FlatHashTable.defaultLoadFactor
    out.writeInt(underlying.size)
    out.writeInt(0) // seedvalue; irrelevant for compatibility writes
    out.writeBoolean(false) // isSizeMapDefined
    underlying.foreach(out.writeObject)
  }

  @throws[java.io.IOException]
  @throws[ClassNotFoundException]
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    underlying = mutable.Set.empty[A]

    // Scala 2.12 FlatHashTable serialization payload. The old load factor/seed/size-map
    // internals are not needed by this compatibility Set implementation, but must be consumed
    // in the exact order to keep the ObjectInputStream aligned.
    in.readInt() // loadFactor
    val size = in.readInt()
    in.readInt() // seedvalue
    in.readBoolean() // isSizeMapDefined

    var index = 0
    while (index < size) {
      underlying += in.readObject().asInstanceOf[A]
      index += 1
    }
  }
}
