package com.johnsnowlabs.nlp.serialization

import java.io.{ObjectInputStream, ObjectOutputStream}

/** Copied from Scala 2.12.
  *
  * @param orig
  */
@SerialVersionUID(212L)
class LegacyListSerializationProxy(@transient private var orig: List[Any]) extends Serializable {

  private def writeObject(out: ObjectOutputStream): Unit = {
    out.defaultWriteObject()
    var xs: List[Any] = orig
    while (xs.nonEmpty) {
      out.writeObject(xs.head)
      xs = xs.tail
    }
    out.writeObject(LegacyListSerializeEnd)
  }

  private def tmpGetObjVal(obj: Any) = {
    obj match {
      case s: String => s"\"$s\""
      case c: Char => s"'$c'"
      case n: Number => n.toString
      case b: Boolean => b.toString
      case null => "null"
      case other => s"instance(${other.getClass.getName})"
    }
  }

  // Java serialization calls this before readResolve during deserialization.
  // Read the whole list and store it in `orig`.
  private def readObject(in: ObjectInputStream): Unit = {
    in.defaultReadObject()
    val builder = List.newBuilder[Any]
    while (true) {
      val obj = in.readObject
      val isListEnd =
        obj.getClass.getName == LegacyListSerializeEnd.getClass.getName
      if (isListEnd) {
        /*        println("DHA: LLSP: reached end of list.")*/
        orig = builder.result()
        return
      } else {
//        println(
//          s"DHA: LLSP: adding to builder list: ${obj.getClass.getName}=${tmpGetObjVal(obj)}")
        builder += obj // original code casts to type, we use Any
      }
    }
  }

  // Provide the result stored in `orig` for Java serialization
  private def readResolve(): AnyRef = {
    /*    println("DHA: List readResolve()!")*/
    orig
  }
}

/** Only used for list serialization */
@SerialVersionUID(212L)
case object LegacyListSerializeEnd
