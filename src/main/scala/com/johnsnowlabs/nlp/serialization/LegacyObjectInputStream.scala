package com.johnsnowlabs.nlp.serialization

import java.io.{ByteArrayInputStream, IOException, ObjectInputStream, ObjectStreamClass}
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/** Custom ObjectInputStream that ignores the serialVersionUID check for a provided class during
  * deserialization.
  *
  * @param in
  *   ByteArrayInputStream of the deserialization
  * @param replacedClass
  *   The class that should be the replacement
  * @param serializedClassName
  *   The name of the serialized class in the input stream
  */
class LegacyObjectInputStream(
    in: ByteArrayInputStream,
    val replacedClass: Class[_],
    val serializedClassName: String)
    extends ObjectInputStream(in) {

  /** Reads the class descriptor from the serialization stream, handling conflicting
    * serialVersionUIDs (SUID) of old Spark NLP objects.
    *
    * We try to read the objects regardless of SUID (by ignoring them). In the case of reading old
    * Maps, we need to use the serialization proxy from Scala 2.12 (removed in 2.13).
    *
    * Taken and adapted from
    * https://stackoverflow.com/questions/795470/how-to-deserialize-an-object-persisted-in-a-db-now-when-the-object-has-different
    * @throws IOException
    *   if an I/O error occurs
    * @throws ClassNotFoundException
    *   if the class of a serialized object could not be found
    *
    * @return
    *   The class descriptor to be used for deserialization.
    */
  @throws[IOException]("I/O error occurred")
  @throws[ClassNotFoundException]("class of a serialized object could not be found")
  override protected def readClassDescriptor: ObjectStreamClass = {
    var resultClassDescriptor = super.readClassDescriptor // initially streams descriptor

    def checkSerializationProxy(): ObjectStreamClass = {
      resultClassDescriptor.getName match {
        case "scala.collection.immutable.HashMap$SerializationProxy" =>
          ObjectStreamClass.lookup(classOf[LegacyHashMapSerializationProxy])
        case "scala.collection.immutable.List$SerializationProxy" =>
          /*          println("DHA: Using LegacyListSerializationProxy")*/
          ObjectStreamClass.lookup(classOf[LegacyListSerializationProxy])
        case "scala.collection.immutable.ListSerializeEnd$" =>
          println("DHA: Using LegacyListSerializationEnd")
          ObjectStreamClass.lookup(LegacyListSerializeEnd.getClass)
        case _ => null // No replacement class found
      }
    }

    // Ignore all serialVersionUIDs, if they are not array
    if (!resultClassDescriptor.getName.startsWith("[")) {
      val classForName = Try {
        Class.forName(resultClassDescriptor.getName, false, getClass.getClassLoader)
      }
      val localClassDescriptor: ObjectStreamClass = classForName match {
        case Success(clazz) => ObjectStreamClass.lookup(clazz)
        case Failure(_) => checkSerializationProxy() // SerializationProxy case
      }

      if (localClassDescriptor != null) {
        val localSUID = localClassDescriptor.getSerialVersionUID
        val streamSUID = resultClassDescriptor.getSerialVersionUID
        if (streamSUID != localSUID) { // check for serialVersionUID mismatch.
          // Use local class descriptor for deserialization
          resultClassDescriptor = localClassDescriptor
        }
      }
    }

    resultClassDescriptor
  }
}

object LegacyObjectInputStream {

  /** Deserialize this class using a custom object input stream, handling serialVersionUID
    * mismatches and loads a replacement class instead. This assumes that the objects were
    * serialized as an array.
    *
    * @param bytes
    *   The bytes to deserialized (read by BytesWritable)
    * @param serializedClassName
    *   The name of the serialized class to replace. By default, chooses the same class as the
    *   type T
    * @tparam T
    *   The type of the array contents, which will be the replacement for serializedClassName
    * @return
    */
  def deserializeArray[T: ClassTag](
      bytes: Array[Byte],
      serializedClassName: Option[String] = None): Array[T] = {
    val bis = new ByteArrayInputStream(bytes)

    // Use ClassTag to store runtime information of class and avoid type erasure.
    // Retrieves the implicitly context-bound parameter of the ClassTag
    val className =
      if (serializedClassName.nonEmpty) serializedClassName.get
      else implicitly[ClassTag[T]].runtimeClass.getCanonicalName

    val ois =
      new LegacyObjectInputStream(bis, implicitly[ClassTag[T]].runtimeClass, className)

    ois.readObject.asInstanceOf[Array[T]]
  }

}
