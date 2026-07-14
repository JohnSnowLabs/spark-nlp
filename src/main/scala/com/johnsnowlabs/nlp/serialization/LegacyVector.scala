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

import scala.collection.mutable.ArrayBuffer

/** Compatibility reader for immutable.Vector instances serialized with Scala 2.12.
  *
  * Scala 2.12 serialized Vector's internal 32-way trie fields directly. Scala 2.13 keeps the
  * class name but has a different implementation and serialVersionUID, so forcing the stream onto
  * the local descriptor can fail with `InvalidClassException: scala.collection.immutable.Vector;
  * unable to create instance`. This shim reads the old fields and resolves to a Scala 2.13
  * Vector.
  */
@SerialVersionUID(-1334388273712300479L)
private[serialization] class LegacyVector[A] extends Serializable {

  private var startIndex: Int = 0
  private var endIndex: Int = 0
  private var focus: Int = 0
  private var depth: Int = 0
  private var dirty: Boolean = false

  private var display0: Array[AnyRef] = _
  private var display1: Array[AnyRef] = _
  private var display2: Array[AnyRef] = _
  private var display3: Array[AnyRef] = _
  private var display4: Array[AnyRef] = _
  private var display5: Array[AnyRef] = _

  private def readResolve(): AnyRef = {
    val expectedSize = math.max(0, endIndex - startIndex)
    val values = ArrayBuffer.empty[A]

    def appendTree(value: Any): Unit = value match {
      case null =>
      case array: Array[_] => array.foreach(appendTree)
      case element => values += element.asInstanceOf[A]
    }

    val root = depth match {
      case 1 => display0
      case 2 => display1
      case 3 => display2
      case 4 => display3
      case 5 => display4
      case 6 => display5
      case _ => display0
    }

    appendTree(root)

    // Old Vector display arrays may contain padded nulls. Keep the logical slice only.
    values.take(expectedSize).toVector
  }
}
