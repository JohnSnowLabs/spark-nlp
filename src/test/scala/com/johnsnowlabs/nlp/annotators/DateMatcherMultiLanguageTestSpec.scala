/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
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

import com.johnsnowlabs.nlp.AnnotatorType.DATE
import com.johnsnowlabs.nlp.{Annotation, AnnotatorType, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

import java.util.Calendar


class DateMatcherMultiLanguageTestSpec extends FlatSpec with DateMatcherBehaviors {

  "a DateMatcher" should "be catching dates" taggedAs FastTest in {

//    val data: Dataset[Row] = DataBuilder.basicDataBuild("I left on the 23/12/2021")
    val data: Dataset[Row] = DataBuilder.basicDataBuild("Sono partito dalla Francia il 29 Marzo")

    data.show(false)

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("dd/MM/yyyy")
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    pipeline.fit(data).transform(data).show(false)
  }

}
