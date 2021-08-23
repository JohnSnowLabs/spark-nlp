/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{explode, size}
import org.scalatest._

class ElmoEmbeddingsTestSpec extends FlatSpec {

  "Elmo Embeddings" should "generate annotations" taggedAs SlowTest in {
    System.out.println("Working Directory = " + System.getProperty("user.dir"))
    val data = Seq(
      "I like pancakes in the summer. I hate ice cream in winter.",
      "If I had asked people what they wanted, they would have said faster horses"
    ).toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val elmoSavedModel = ElmoEmbeddings.pretrained()
      .setPoolingLayer("word_emb")
      .setInputCols(Array("token", "document"))
      .setOutputCol("embeddings")

    elmoSavedModel.write.overwrite().save("./tmp_elmo_tf")

    val embeddings = ElmoEmbeddings.load("./tmp_elmo_tf")

    val pipeline = new Pipeline().setStages(Array(
      document,
      sentence,
      tokenizer,
      embeddings
    ))

    val elmoDDD = pipeline.fit(data).transform(data)

    elmoDDD.select("embeddings.result").show(false)
    elmoDDD.select("embeddings.metadata").show(false)
    val explodeEmbds = elmoDDD.select(explode($"embeddings.embeddings").as("embedding"))
    elmoDDD.select(size(elmoDDD("embeddings.embeddings")).as("embeddings_size")).show
    explodeEmbds.select(size($"embedding").as("embeddings_size")).show

  }
}
