/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp

import com.johnsnowlabs.tags.{FastTest, SlowTest}
import com.johnsnowlabs.util.ConfigHelper.{awsJavaSdkVersion, hadoopAwsVersion}
import org.apache.spark.sql.functions.col
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

  it should "structure HTML files" taggedAs FastTest in {
    val htmlFilePath = "./src/test/resources/reader/html/example-10k.html"
    val htmlDF = SparkNLP.read.html(htmlFilePath)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "structured HTML files with params" taggedAs FastTest in {
    val htmlFilePath = "./src/test/resources/reader/html/example-10k.html"
    val params = Map("titleFontSize" -> "10")
    val htmlDF = SparkNLP.read(params).html(htmlFilePath)
    htmlDF.show()

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "structured HTML in real time" taggedAs SlowTest in {
    val url = "https://www.wikipedia.org"
    val htmlDF = SparkNLP.read.html(url)

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "structured HTML in real time for a set of URLs" taggedAs SlowTest in {
    val urls = Array("https://www.wikipedia.org", "https://example.com/")
    val htmlDF = SparkNLP.read.html(urls)

    assert(!htmlDF.select(col("html").getItem(0)).isEmpty)
  }

  it should "rise an exception when HTML file is invalid" taggedAs FastTest in {
    val htmlFilePath = ""

    val htmlDF = SparkNLP.read.html(htmlFilePath)

    assertThrows[Exception] {
      htmlDF.show()
    }
  }

  it should "structured Email files" taggedAs FastTest in {
    val emailDirectory = "src/test/resources/reader/email"
    val emailDF = SparkNLP.read.email(emailDirectory)

    assert(!emailDF.select(col("email").getItem(0)).isEmpty)
  }

}
