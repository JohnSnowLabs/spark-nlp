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

import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorType}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


trait LemmatizerBehaviors { this: FlatSpec =>

  def fullLemmatizerPipeline(dataset: => Dataset[Row]) {
    "a Lemmatizer Annotator" should "succesfully transform data" taggedAs FastTest in {
      dataset.show
      AnnotatorBuilder.withFullLemmatizer(dataset)
        .collect().foreach {
        row =>
          row.getSeq[Row](2)
            .map(Annotation(_))
            .foreach {
              case lemma: Annotation if lemma.annotatorType == AnnotatorType.TOKEN =>
                println(lemma, lemma.result)
              case _ => ()
            }
      }
    }
  }
}
