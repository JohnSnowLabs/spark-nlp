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
  * @param resolveCustomDescriptor
  *   Method to provide additional legacy class descriptor mappings for custom classes. By
  *   default, doesn't provide additional classes.
  */
class LegacyObjectInputStream(
    in: ByteArrayInputStream,
    val replacedClass: Class[_],
    val serializedClassName: String,
    resolveCustomDescriptor: ObjectStreamClass => ObjectStreamClass)
    extends ObjectInputStream(in) {

  /** Checks for explicit mappings of old serialized class names to replacement classes such as
    * Scala 2.12 collection serialization proxies and Spark NLP legacy classes.
    *
    * @param classDescriptor
    *   The class descriptor read from the deserialization stream
    * @return
    *   The replacement class descriptor, or null if no mapping found
    */
  protected def resolveLegacyDescriptor(classDescriptor: ObjectStreamClass): ObjectStreamClass = {
    classDescriptor.getName match {
      case "scala.collection.immutable.HashMap$SerializationProxy" =>
        ObjectStreamClass.lookup(classOf[LegacyHashMapSerializationProxy])
      case "scala.collection.immutable.HashSet$SerializationProxy" =>
        ObjectStreamClass.lookup(classOf[LegacyHashSetSerializationProxy])
      case "scala.collection.mutable.HashSet" =>
        ObjectStreamClass.lookup(classOf[LegacyMutableHashSet[_]])
      case "scala.collection.mutable.HashMap" =>
        ObjectStreamClass.lookup(classOf[LegacyMutableHashMap[_, _]])
      case "scala.collection.immutable.Vector" =>
        ObjectStreamClass.lookup(classOf[LegacyVector[_]])
      case "scala.collection.immutable.List$SerializationProxy" =>
        ObjectStreamClass.lookup(classOf[LegacyListSerializationProxy])
      case "scala.collection.immutable.ListSerializeEnd$" =>
        ObjectStreamClass.lookup(LegacyListSerializeEnd.getClass)
      case _ => // No replacement class found, delegate to subclass
        resolveCustomDescriptor(classDescriptor)
    }
  }

  /** Reads the class descriptor from the serialization stream, handling conflicting
    * serialVersionUIDs (SUID) of old Spark NLP objects.
    *
    * We try to read the objects regardless of SUID (by ignoring them). In the case of reading old
    * Maps, we need to use the serialization proxy from Scala 2.12 (removed in 2.13).
    *
    * Taken and adapted from
    * https://stackoverflow.com/questions/795470/how-to-deserialize-an-object-persisted-in-a-db-now-when-the-object-has-different
    *
    * @throws IOException
    *   if an I/O error occurs
    * @throws ClassNotFoundException
    *   if the class of a serialized object could not be found
    * @return
    *   The class descriptor to be used for deserialization.
    */
  @throws[IOException]("I/O error occurred")
  @throws[ClassNotFoundException]("class of a serialized object could not be found")
  override protected def readClassDescriptor: ObjectStreamClass = {
    var resultClassDescriptor = super.readClassDescriptor // initially streams descriptor

    // Ignore all serialVersionUIDs, if they are not array
    if (!resultClassDescriptor.getName.startsWith("[")) {
      val legacyClassDescriptor = resolveLegacyDescriptor(resultClassDescriptor)
      val classForName = Try {
        Class.forName(resultClassDescriptor.getName, false, getClass.getClassLoader)
      }
      val localClassDescriptor: ObjectStreamClass =
        if (legacyClassDescriptor != null) legacyClassDescriptor
        else
          classForName match {
            case Success(clazz) => ObjectStreamClass.lookup(clazz)
            case Failure(_) => null
          }

      if (localClassDescriptor != null) {
        val localSUID = localClassDescriptor.getSerialVersionUID
        val streamSUID = resultClassDescriptor.getSerialVersionUID
        if (legacyClassDescriptor != null || streamSUID != localSUID) { // check for explicit legacy mapping or serialVersionUID mismatch.
          // Use local class descriptor for deserialization
          resultClassDescriptor = localClassDescriptor
        }
      }
    }

    resultClassDescriptor
  }

  /** Resolves a class after [[readClassDescriptor]] during deserialization, trying first with the
    * thread context classloader (Spark separates loaders), then falling back to default behavior.
    *
    * @param desc
    *   The class descriptor read from the deserialization stream
    * @return
    *   The resolved class
    */
  @throws[IOException]
  @throws[ClassNotFoundException]
  override protected def resolveClass(desc: ObjectStreamClass): Class[_] = {
    try {
      // Try with thread context classloader first (Spark workaround)
      Class.forName(desc.getName, false, Thread.currentThread().getContextClassLoader)
    } catch {
      case _: ClassNotFoundException =>
        // Fallback to default behavior
        super.resolveClass(desc)
    }
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
    * @param resolveCustomDescriptor
    *   Method to provide additional legacy class descriptor mappings for custom classes.
    * @tparam T
    *   The type of the array contents, which will be the replacement for serializedClassName
    * @return
    */
  def deserializeArray[T: ClassTag](
      bytes: Array[Byte],
      serializedClassName: Option[String] = None,
      resolveCustomDescriptor: ObjectStreamClass => ObjectStreamClass = (_: ObjectStreamClass) =>
        null): Array[T] = {
    val bis = new ByteArrayInputStream(bytes)

    // Use ClassTag to store runtime information of class and avoid type erasure.
    // Retrieves the implicitly context-bound parameter of the ClassTag
    val className =
      if (serializedClassName.nonEmpty) serializedClassName.get
      else implicitly[ClassTag[T]].runtimeClass.getCanonicalName

    val ois =
      new LegacyObjectInputStream(
        bis,
        implicitly[ClassTag[T]].runtimeClass,
        className,
        resolveCustomDescriptor)

    ois.readObject.asInstanceOf[Array[T]]
  }

}
