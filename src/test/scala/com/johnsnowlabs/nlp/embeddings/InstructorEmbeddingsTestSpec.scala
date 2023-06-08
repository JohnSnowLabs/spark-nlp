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
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class InstructorEmbeddingsTestSpec extends AnyFlatSpec {

  "Instructor Embeddings" should "correctly embed multiple sentences" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    val ddd = Seq(
      "Capitalism has been dominant in the Western world since the end of feudalism, but most feel[who?]" +
        " that the term \"mixed economies\" more precisely describes most contemporary economies, due to their " +
        "containing both private-owned and state-owned enterprises. In capitalism, prices determine the " +
        "demand-supply scale. For example, higher demand for certain goods and services lead to higher prices " +
        "and lower demand for certain goods lead to lower prices.",
      "The disparate impact theory is especially controversial under the Fair Housing Act because the Act " +
        "regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars" +
        " have argued that the theory's use under the Fair Housing Act, combined with extensions of the " +
        "Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. " +
        "housing market and ensuing global economic recession",
      "Disparate impact in United States labor law refers to practices in employment, housing, and other" +
        " areas that adversely affect one group of people of a protected characteristic more than another, " +
        "even though rules applied by employers or landlords are formally neutral. Although the protected classes " +
        "vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, " +
        "and sex as protected traits, and some laws include disability status and other traits as well.")
      .toDF("text")

    val document = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val embeddings = InstructorEmbeddings
      .pretrained()
      .setInstruction("Represent the Wikipedia document for retrieval: ")
      .setInputCols(Array("document"))
      .setOutputCol("instructor")

    val pipeline = new Pipeline().setStages(Array(document, embeddings))

    val pipelineDF = pipeline.fit(ddd).transform(ddd)
    pipelineDF.select("instructor.embeddings").show(truncate = false)

  }
}
