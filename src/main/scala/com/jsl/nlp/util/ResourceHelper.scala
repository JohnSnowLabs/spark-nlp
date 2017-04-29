package com.jsl.nlp.util

import scala.io.Source

import scala.collection.mutable.{Map => MMap}
/**
  * Created by saif on 28/04/17.
  */
object ResourceHelper {

  /**
    * Standard key value parser from source
    *
    * @param source File input to streamline
    * @param format format, for now only txt
    * @param keySep separator cha
    * @param valueSep values separator in dictionary
    * @return
    */
  def parseKeyValueText(
                         source: Source,
                         format: String,
                         keySep: String,
                         valueSep: String): Map[String, Array[String]] = {
    val source = Source.fromFile("/home/saif/readtest/src/main/resources/input.txt")
    val res = source.getLines.map(line => {
      val kv = line.split(keySep).map(_.trim)
      val key = kv(0)
      val values = kv(1).split(valueSep).map(_.trim)
      (key, values)
    }).toMap
    source.close()
    res
  }

  /**
    * Specific approach chosen is to generate quick reads of Lemma Dictionary
    *
    * @param source File input to streamline
    * @param format format, for now only txt
    * @param keySep separator cha
    * @param valueSep values separator in dictionary
    * @return
    */
  def flattenValuesAsKeys(
                                 source: Source,
                                 format: String,
                                 keySep: String,
                                 valueSep: String): Map[String, String] = {
    format match {
      case "txt" => {
        val m: MMap[String, String] = MMap()
        source.getLines.foreach( line => {
          val kv = line.split(keySep).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(valueSep).map(_.trim)
          values.foreach(m(_) = key)
        })
        source.close()
        m.toMap
      }
      case _ => throw new IllegalArgumentException("Only txt supported as a file format")
    }
  }

}
