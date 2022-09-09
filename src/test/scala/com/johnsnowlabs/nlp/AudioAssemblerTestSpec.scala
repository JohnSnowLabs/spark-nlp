package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class AudioAssemblerTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark

  it should "run in a LightPipeline" taggedAs FastTest in {}

}
