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

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.SparkAccessor
import org.scalatest._
import com.johnsnowlabs.tags.{FastTest, SlowTest}

class ExportCSVToolTestSpec extends FlatSpec with ExportCSVToolBehaviors {

  import SparkAccessor.spark.implicits._

  System.gc()

  val document = Seq(
    "EU rejects German call to boycott British lamb",
    "Peter Blackburn"
  ).toDS.toDF("text")

  val path = "tmp/tmp.txt"
  val data = SparkAccessor.spark.sparkContext.wholeTextFiles(path).toDS.toDF("filename", "text")
  "a POS tagger annotator" should behave like testExportToCoNLLFile(data, "tmp/tmp.csv")

  "a CVS export for multiple files" should behave like
    testExportSeveralCoNLLFiles("tmp")

  val document2 = Seq(
    (Array("Peter", "Blackburn", "Another", "sentence"),
    Array("NNP", "NNP", "NN", "NN"),
    Array("B-something", "I-something", "B-something", "I-something"),
    Array(("sentence","1"), ("sentence","1"),
      ("sentence","2"), ("sentence","2")))
  ).toDS.toDF("token", "pos", "result", "metadata")

  "an evaluation" should behave like testEvaluation(document2)

}
