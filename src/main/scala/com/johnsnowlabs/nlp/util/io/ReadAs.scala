package com.johnsnowlabs.nlp.util.io

object ReadAs extends Enumeration {
  implicit def str2frmt(v: String): Format = {
    v.toUpperCase match {
      case "LINE_BY_LINE" => LINE_BY_LINE
      case "SPARK_DATASET" => SPARK_DATASET
      case _ => throw new MatchError("ReadAs must be either LINE_BY_LINE or SPARK_DATASET")
    }
  }
  type Format = Value
  val LINE_BY_LINE, SPARK_DATASET = Value
}
