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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder, _}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class NormalizerTestSpec extends FlatSpec with NormalizerBehaviors {

  "A normalizer" should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    val normalizer = new Normalizer
    assert(normalizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullNormalizerPipeline(latinBodyData)
  "A Normalizer pipeline with latin content and disabled lowercasing" should behave like lowercasingNormalizerPipeline(latinBodyData)

  private var data = Seq(
    ("lol", "laugh@out@loud"),
    ("gr8", "great"),
     ("b4", "before"),
    ("4", "for"),
    ("Yo dude whatsup?", "hey@friend@how@are@you")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testCorrectSlangs(data)

  data = Seq(
    ("test-ing", "testing"),
    ("test-ingX", "testing")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testMultipleRegexPatterns(data)

}
