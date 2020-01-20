package com.johnsnowlabs.nlp.util.io

object ReadAs extends Enumeration {
  implicit def str2frmt(v: String): Format = {
    v.toUpperCase match {
      case "SPARK" => SPARK
      case "TEXT" => TEXT
      case "BINARY" => BINARY
      case _ => throw new MatchError(s"Invalid ReadAs. Must be either of ${this.values.mkString("|")}")
    }
  }
  type Format = Value
  val SPARK, TEXT, BINARY = Value
}
