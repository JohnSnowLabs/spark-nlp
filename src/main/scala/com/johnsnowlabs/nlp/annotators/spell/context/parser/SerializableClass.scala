package com.johnsnowlabs.nlp.annotators.spell.context.parser

import java.io.{ObjectInputStream, ObjectOutputStream}
import com.esotericsoftware.kryo.KryoSerializable
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.liblevenshtein.proto.LibLevenshteinProtos.DawgNode
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer, Transducer}
import com.esotericsoftware.kryo.Kryo

trait SerializableClass extends Serializable with KryoSerializable {
  this:SpecialClassParser =>

  // these are for standard Java serialization
  def deserializeTransducer(aInputStream:ObjectInputStream) = {
    aInputStream.defaultReadObject()
    val serializer = new PlainTextSerializer
    val size = aInputStream.readInt()
    val bytes = new Array[Byte](size)
    aInputStream.readFully(bytes)
    serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
  }

  def serializeTransducer(aOutputStream:ObjectOutputStream, t:ITransducer[Candidate])= {
    aOutputStream.defaultWriteObject()
    val serializer = new PlainTextSerializer
    val transBytes = serializer.serialize(t)
    aOutputStream.writeInt(transBytes.length)
    aOutputStream.write(transBytes)
  }

  // these are for Kryo serialization
  def write(kryo: Kryo, output: Output): Unit = {
    val serializer = new PlainTextSerializer
    val transBytes = serializer.serialize(transducer)
    output.writeInt(transBytes.length)
    output.write(transBytes)
  }

  def read(kryo: Kryo, input: Input): Unit = {
    val serializer = new PlainTextSerializer
    val size = input.readInt()
    val bytes = new Array[Byte](size)
    input.read(bytes)
    transducer = serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
  }
}
