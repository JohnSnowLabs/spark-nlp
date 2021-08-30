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

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.Row
import org.scalatest.flatspec.AnyFlatSpec

class CombinedTestSpec extends AnyFlatSpec {

  "Simple combined annotators" should "successfully go through all transformations" taggedAs FastTest in {
    val data = DataBuilder.basicDataBuild("This is my first sentence. This is your second list of words")
    val transformation = AnnotatorBuilder.withLemmaTaggedSentences(data)
    transformation
      .collect().foreach {
      row =>
        row.getSeq[Row](1).map(Annotation(_)).foreach { token =>
          // Document annotation
          assert(token.annotatorType == DOCUMENT)
        }
        row.getSeq[Row](2).map(Annotation(_)).foreach { token =>
          // SBD annotation
          assert(token.annotatorType == DOCUMENT)
        }
        row.getSeq[Row](4).map(Annotation(_)).foreach { token =>
          // POS annotation
          assert(token.annotatorType == POS)
        }
    }
  }
}
