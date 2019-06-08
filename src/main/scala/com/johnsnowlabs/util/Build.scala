package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper

object Build {
  val version: String = {
    val objPackage = ResourceHelper.getClass.getPackage
    val version = objPackage.getSpecificationVersion

    // When spark-nlp library is a jar
    if (version != null && version.nonEmpty)
      version
    else
      "2.0.8"
  }
}