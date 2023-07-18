package com.johnsnowlabs.nlp

import com.johnsnowlabs.tags.FastTest
import com.johnsnowlabs.util.ConfigHelper.{awsJavaSdkVersion, hadoopAwsVersion}
import org.scalatest.flatspec.AnyFlatSpec

class SparkNLPTestSpec extends AnyFlatSpec {

  behavior of "SparkNLPTestSpec"

  it should "start with extra parameters" taggedAs FastTest in {
    val extraParams: Map[String, String] = Map(
      "spark.jars.packages" -> ("org.apache.hadoop:hadoop-aws:" + hadoopAwsVersion + ",com.amazonaws:aws-java-sdk:" + awsJavaSdkVersion),
      "spark.hadoop.fs.s3a.path.style.access" -> "true",
      "spark.driver.cores" -> "2")

    val spark = SparkNLP.start(params = extraParams)

    assert(spark.conf.get("spark.hadoop.fs.s3a.path.style.access") == "true")
    assert(spark.conf.get("spark.master") == "local[2]")

    Seq(
      "com.johnsnowlabs.nlp:spark-nlp",
      "org.apache.hadoop:hadoop-aws",
      "com.amazonaws:aws-java-sdk").foreach { pkg =>
      assert(spark.conf.get("spark.jars.packages").contains(pkg))
    }
  }
}
