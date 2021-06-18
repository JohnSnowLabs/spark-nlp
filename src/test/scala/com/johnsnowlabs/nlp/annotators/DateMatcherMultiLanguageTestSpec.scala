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

import com.johnsnowlabs.nlp.{Annotation, DataBuilder}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class DateMatcherMultiLanguageTestSpec extends FlatSpec with DateMatcherBehaviors {

  "a DateMatcher" should "be catching formatted italian dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Sono arrivato in Francia il 15/9/2012.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "09/15/2012")
  }

  "a DateMatcher" should "be catching unformatted italian dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Sono arrivato in Francia il 15 Settembre 2012.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "09/15/2012")
  }

  "a DateMatcher" should "be catching formatted french dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Je suis arrivé en France le 23/5/2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("fr")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "05/23/2019")
  }

  "a DateMatcher" should "be catching unformatted french dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Je suis arrivé en France le 23 mai 2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("fr")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "05/23/2019")
  }

  "a DateMatcher" should "be catching formatted portuguese dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Cheguei à França no dia 23/5/2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("pt")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "05/23/2019")
  }

  "a DateMatcher" should "be catching unformatted portuguese dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Cheguei à França em 23 de maio de 2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)
      .setSourceLanguage("pt")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "05/23/2019")
  }

  "a DateMatcher" should "be catching unspecified portuguese language dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Cheguei à França em 23 de maio de 2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "05/23/2019")
  }

  "a DateMatcher" should "be catching unspecified french language dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Je suis arrivé en France le 23 février 2019.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "02/23/2019")
  }

  "a DateMatcher" should "be catching unspecified italian language dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild("Sono arrivato in Francia il 15 Settembre 2012.")

    val dateMatcher = new DateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setMultiLanguageCapability(true)

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations.head.result == "09/15/2012")
  }

}
