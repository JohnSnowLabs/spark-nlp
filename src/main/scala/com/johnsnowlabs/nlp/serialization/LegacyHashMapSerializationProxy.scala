package com.johnsnowlabs.nlp.serialization

import scala.collection.immutable.HashMap
import scala.collection.immutable.HashMap.empty

/** Serialization Proxy to deserialize HashMaps the Scala 2.12 way.
  *
  * @param orig
  *   HashMap to init
  */
@SerialVersionUID(212L)
class LegacyHashMapSerializationProxy(@transient private var orig: HashMap[Any, Any])
    extends Serializable {
  private def writeObject(out: java.io.ObjectOutputStream): Unit = {
    val s = orig.size
    out.writeInt(s)
    for ((k, v) <- orig) {
      out.writeObject(k)
      out.writeObject(v)
    }
  }

  private def readObject(in: java.io.ObjectInputStream): Unit = {
    orig = empty
    val s = in.readInt()
    for (_ <- 0 until s) {
      // In original implementation this was cast to K, V type
      // But reading it as Any also seems to work for our case (implicit casting later on)
      // TODO: We might want to check if this is an issue anywhere.
      val key = in.readObject()
      val value = in.readObject()
      orig = orig.updated(key, value)
    }
  }

  private def readResolve(): AnyRef = orig
}
