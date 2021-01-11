package com.johnsnowlabs.nlp.annotators.spell.context.parser

import java.io.{IOException, ObjectInputStream, ObjectOutputStream}

import com.github.liblevenshtein.proto.LibLevenshteinProtos.DawgNode
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer, Transducer}

trait SerializableClass extends Serializable{

  def deserializeTransducer(aInputStream:ObjectInputStream) = {
    aInputStream.defaultReadObject()
    val serializer = new PlainTextSerializer
    val size = aInputStream.readInt()
    val bytes = new Array[Byte](size)
    aInputStream.readFully(bytes)
    serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], bytes)
  }


  def serializeTransducer(aOutputStream:ObjectOutputStream, transducer:ITransducer[Candidate])= {
    aOutputStream.defaultWriteObject()
    val serializer = new PlainTextSerializer
    val transBytes = serializer.serialize(transducer)
    aOutputStream.writeInt(transBytes.length)
    aOutputStream.write(transBytes)
  }



}
