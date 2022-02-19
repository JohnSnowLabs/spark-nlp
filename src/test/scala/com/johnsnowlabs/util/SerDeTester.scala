package com.johnsnowlabs.util

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import java.nio.file.{Files, Paths}


@SerialVersionUID(136630359430337L)
class MyTestObj(val name: String) extends Serializable {
  override def toString = name
  val greet: String = {
    s"Hello, $name"
  }
}


object SerializationDemoWriter extends App {

  private def serializeMyTestObj = {
    val dirPath = "tmp"
    Files.createDirectories(Paths.get(dirPath))
    val oos = new ObjectOutputStream(new FileOutputStream(dirPath.concat("/").concat("myTestObjInScala213")))
    oos.writeObject(myTestObj)
    oos.close
    println(java.io.ObjectStreamClass.lookup(myTestObj.getClass()).getSerialVersionUID())
  }

  val myTestObj = new MyTestObj("Stef")
  serializeMyTestObj
  println(myTestObj)
}

object SerializationDemoReader extends App {

  private def deserializeMyTestObj = {
    val dirPath = "tmp"
    val ois = new ObjectInputStream(new FileInputStream(dirPath.concat("/").concat("myTestObjInScala212")))
    val myTestObj = ois.readObject.asInstanceOf[MyTestObj]
    ois.close

    myTestObj
  }
  val deserialized = deserializeMyTestObj
  println(s"$deserialized with serial version UID ")
  println(java.io.ObjectStreamClass.lookup(deserialized.getClass()).getSerialVersionUID())
}