package com.johnsnowlabs.nlp.annotators.spell.context
import com.esotericsoftware.kryo.Serializer
import com.esotericsoftware.kryo.Kryo
import com.esotericsoftware.kryo.io.{Input, Output}
import com.github.liblevenshtein.proto.LibLevenshteinProtos.DawgNode
import com.github.liblevenshtein.serialization.PlainTextSerializer
import com.github.liblevenshtein.transducer.{Candidate, ITransducer, Transducer}
import org.apache.spark.serializer.KryoRegistrator

class ContextSpellRegistrator extends KryoRegistrator {
  override def registerClasses(kryo: Kryo) {
    kryo.register(classOf[Transducer[DawgNode, Candidate]], new CustomTransducerSerializer())
  }
}

class CustomTransducerSerializer extends Serializer[Transducer[DawgNode, Candidate]] {

  override def write(kryo: Kryo, output: Output, t: Transducer[DawgNode, Candidate]): Unit = {
    val serializer = new PlainTextSerializer
    serializer.serialize(t, output)
  }

  override def read(kryo: Kryo, input: Input, classType: Class[Transducer[DawgNode, Candidate]]): Transducer[DawgNode, Candidate] = {
    val serializer = new PlainTextSerializer
    serializer.deserialize(classOf[Transducer[DawgNode, Candidate]], input)
  }
}

