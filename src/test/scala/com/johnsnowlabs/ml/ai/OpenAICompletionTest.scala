/*
 * Copyright 2017-2023 John Snow Labs
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
package com.johnsnowlabs.ml.ai

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class OpenAICompletionTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  "OpenAICompletion" should "generate a completion for prompts" taggedAs SlowTest in {
    // Set OPENAI_API_KEY env variable to make this work
    val promptDF = Seq(
      "Generate a restaurant review.",
      "Write a review for a local eatery.",
      "Create a JSON with a review of a dining experience.").toDS.toDF("text")

    promptDF.show(false)

    val openAICompletion = new OpenAICompletion()
      .setInputCols("document")
      .setOutputCol("completion")
      .setModel("text-davinci-003")
      .setMaxTokens(50)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, openAICompletion))
    val completionDF = pipeline.fit(promptDF).transform(promptDF)
    completionDF.select("completion").show(false)
  }

}
