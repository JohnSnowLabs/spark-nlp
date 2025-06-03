package com.johnsnowlabs.nlp.serialization

import scala.collection.immutable.HashSet

/** Copied from Scala 2.12.
  *
  * @param orig
  */
@SerialVersionUID(212L)
private class LegacyHashSetSerializationProxy(@transient private var orig: HashSet[Any])
    extends Serializable {
  private def writeObject(out: java.io.ObjectOutputStream): Unit = {
    val s = orig.size
    out.writeInt(s)
    for (e <- orig) {
      out.writeObject(e)
    }
  }

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    orig = HashSet.empty
    val s = in.readInt()
    for (i <- 0 until s) {
      val e = in.readObject()
      orig = orig + e
    }
  }

  private def readResolve(): AnyRef = orig
}
