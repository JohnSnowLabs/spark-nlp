package com.johnsnowlabs.nlp

import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.ConfigHelper.{awsJavaSdkVersion, hadoopAwsVersion}
import org.scalatest.flatspec.AnyFlatSpec

class SparkNLPTestSpec extends AnyFlatSpec {

  behavior of "SparkNLPTestSpec"

  it should "start with extra parameters" taggedAs SlowTest ignore {
    val extraParams: Map[String, String] = Map(
      "spark.jars.packages" -> ("org.apache.hadoop:hadoop-aws:" + hadoopAwsVersion + ",com.amazonaws:aws-java-sdk:" + awsJavaSdkVersion),
      "spark.hadoop.fs.s3a.path.style.access" -> "true")

    val spark = SparkNLP.start(params = extraParams)

    assert(spark.conf.get("spark.hadoop.fs.s3a.path.style.access") == "true")

    Seq(
      "com.johnsnowlabs.nlp:spark-nlp",
      "org.apache.hadoop:hadoop-aws",
      "com.amazonaws:aws-java-sdk").foreach { pkg =>
      assert(spark.conf.get("spark.jars.packages").contains(pkg))
    }
  }
}
