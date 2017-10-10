package com.jsl.nlp.util.io

object ResourceFormat extends Enumeration {
  implicit def str2frmt(v: String): Format = {
    if (v.toUpperCase == TXT.toString) TXT
    else if (v.toUpperCase == TXTDS.toString) TXTDS
    else if (v.toUpperCase == PARQUET.toString) PARQUET
    else throw new Exception("Unsupported format type")
  }
  type Format = Value
  val TXT, TXTDS, PARQUET = Value
}
