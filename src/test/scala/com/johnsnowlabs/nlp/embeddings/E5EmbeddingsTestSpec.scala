/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class E5EmbeddingsTestSpec extends AnyFlatSpec {

  "E5 Embeddings" should "correctly embed multiple sentences" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "query: how much protein should a female eat",
      "query: summit define",
      "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 " +
        "grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or" +
        " training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
      "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of" +
        " a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more" +
        " governments.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = E5Embeddings
      .pretrained()
      .setInputCols(Array("document"))
      .setOutputCol("e5")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("e5.embeddings").show(truncate = false)

  }
}
