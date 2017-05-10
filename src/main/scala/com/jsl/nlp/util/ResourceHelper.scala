package com.jsl.nlp.util

import java.io.FileNotFoundException

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
                         source: String,
                         format: String,
                         keySep: String,
                         valueSep: String): Map[String, Array[String]] = {
    format match {
      case "txt" =>
        val stream = try {
          getClass.getResourceAsStream("/" + source)
        } catch {
          case _: FileNotFoundException =>
            throw new FileNotFoundException(s"Lemma dictionary $source not found")
        }
        val res = Source.fromInputStream (stream).getLines.map (line => {
          val kv = line.split (keySep).map (_.trim)
          val key = kv (0)
          val values = kv (1).split (valueSep).map (_.trim)
          (key, values)
        }).toMap
        stream.close()
        res
    }
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
                                 source: String,
                                 format: String,
                                 keySep: String,
                                 valueSep: String): Map[String, String] = {
    format match {
      case "txt" => {
        val m: MMap[String, String] = MMap()
        val stream = try {
          getClass.getResourceAsStream("/" + source)
        } catch {
          case _: FileNotFoundException =>
            throw new FileNotFoundException(s"Lemma dictionary $source not found")
        }
        Source.fromInputStream(stream).getLines.foreach( line => {
          val kv = line.split(keySep).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(valueSep).map(_.trim)
          values.foreach(m(_) = key)
        })
        stream.close()
        m.toMap
      }
      case _ => throw new IllegalArgumentException("Only txt supported as a file format")
    }
  }

}
