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

import java.time.format.DateTimeFormatter
import java.time.temporal.TemporalAdjusters
import java.time.{DayOfWeek, LocalDate}


class MultiDateMatcherMultiLanguageTestSpec extends FlatSpec with DateMatcherBehaviors {

  private def getOneDayAgoDate(pattern: String = "MM/dd/yyyy") = {
    val localDate = LocalDate.now.minusDays(1L)
    val formatter = DateTimeFormatter.ofPattern(pattern)
    localDate.format(formatter)
  }

  private def getTwoDaysAgoDate(pattern: String = "MM/dd/yyyy") = {
    val localDate = LocalDate.now.minusDays(2L)
    val formatter = DateTimeFormatter.ofPattern(pattern)
    localDate.format(formatter)
  }

  private def getNextWeekDate(pattern: String = "MM/dd/yyyy") = {
    val localDate = LocalDate.now.plusWeeks(1L)
    val formatter = DateTimeFormatter.ofPattern(pattern)
    localDate.format(formatter)
  }

  private def getInTwoWeeksDate(pattern: String = "MM/dd/yyyy") = {
    val localDate = LocalDate.now.plusWeeks(2L)
    val formatter = DateTimeFormatter.ofPattern(pattern)
    localDate.format(formatter)
  }

  private def getNextWeekOfDay(dof: DayOfWeek, pattern: String = "MM/dd/yyyy") = {
    val localDate = LocalDate.now.`with`(TemporalAdjusters.next(dof))
    val formatter = DateTimeFormatter.ofPattern(pattern)
    localDate.format(formatter)
  }

  "a DateMatcher" should "be catching multiple unformatted english dates using the same factory2" taggedAs FastTest in {

    val data = DataBuilder.basicDataBuild(
      "We met on the 13/5/2018 and then on the 18/5/2020.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations(0).result == "05/13/2018")
    assert(annotations(1).result == "05/18/2020")
  }

  "a DateMatcher" should "be catching multiple unformatted future english dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "I see you next Friday after the next Thursday.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getNextWeekOfDay(DayOfWeek.THURSDAY))
      && results.contains(getNextWeekOfDay(DayOfWeek.FRIDAY))
    )
  }

  "a DateMatcher" should "be catching multiple unformatted english dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "I saw him 2 days ago and he told me that he will visit us next week.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    assert(annotations(0).result == getTwoDaysAgoDate())
    assert(annotations(1).result == getNextWeekDate())
  }

  /** ITALIAN **/

  "a DateMatcher" should "be catching multiple formatted italian dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Ci siamo incontrati il 13/5/2018 e poi il 18/5/2020.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains("05/13/2018") && results.contains("05/18/2020"))
  }

  "a DateMatcher" should "be catching -2 d and +1w unformatted italian dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Ci siamo incontrati 2 giorni fa e mi disse che ci avrebbe visitato la settimana prossima.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getTwoDaysAgoDate()) && results.contains(getNextWeekDate()))
  }

  "a DateMatcher" should "be catching -1 d and +2w unformatted italian dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "L'ho incontrato ieri e mi disse che ci avrebbe visitato tra 2 settimane.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("it")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getOneDayAgoDate()) && results.contains(getInTwoWeeksDate()))
  }

  /** FRENCH **/

  "a DateMatcher" should "be catching multiple formatted french dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Nous nous sommes rencontrés le 13/05/2018 puis le 18/05/2020.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("fr")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains("05/13/2018") && results.contains("05/18/2020"))
  }

  "a DateMatcher" should "be catching -2 d and +1w unformatted french dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Nous nous sommes rencontrés il y a 2 jours " +
        "et il m'a dit qu'il nous rendrait visite la semaine prochaine.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("fr")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getTwoDaysAgoDate()) && results.contains(getNextWeekDate()))
  }

  "a DateMatcher" should "be catching -1 d and +2w unformatted french dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Je l'ai rencontré hier et il m'a dit qu'il nous rendrait visite dans 2 semaines.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("fr")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getOneDayAgoDate()) && results.contains(getInTwoWeeksDate()))
  }

  /** PORTUGUESE **/

  "a DateMatcher" should "be catching multiple formatted portuguese dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Encontramo-nos no dia 13/05/2018 e depois no dia 18/05/2020.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("pt")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains("05/13/2018") && results.contains("05/18/2020"))
  }

  "a DateMatcher" should "be catching -2 d and +1w unformatted portuguese dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Nós nos conhecemos há 2 dias e ele me disse que nos visitaria na próxima semana.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("pt")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getTwoDaysAgoDate()) && results.contains(getNextWeekDate()))
  }

  "a DateMatcher" should "be catching -1 d and +2w unformatted portuguese dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Eu o conheci ontem e ele me disse que nos visitaria em 2 semanas.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("pt")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getOneDayAgoDate()) && results.contains(getInTwoWeeksDate()))
  }

  /** SPANISH **/

  "a DateMatcher" should "be catching multiple formatted spanish dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Nos conocimos el 13/05/2018 y luego el 18/05/2020.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("es")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains("05/13/2018") && results.contains("05/18/2020"))
  }

  "a DateMatcher" should "be catching -2d and +1w unformatted spanish dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Nos conocimos hace 2 días y me dijeron que nos visitaría la semana que viene.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("es")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getTwoDaysAgoDate()) && results.contains(getNextWeekDate()))
  }

  "a DateMatcher" should "be catching -1d and +2w unformatted spanish dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Lo conocí ayer y me dijo que nos visitará dentro de 2 semanas.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("es")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getOneDayAgoDate()) && results.contains(getInTwoWeeksDate()))
  }

  /** GERMAN **/

  "a DateMatcher" should "be catching multiple formatted german dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Wir trafen uns am 13/05/2018 und dann am 18/05/2020 .")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("de")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains("05/13/2018") && results.contains("05/18/2020"))
  }

  "a DateMatcher" should "be catching -2d and +1w unformatted german dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Wir haben uns vor 2 tagen kennengelernt und sie sagten mir, dass sie uns nächste woche besuchen würden.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("de")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getTwoDaysAgoDate()) && results.contains(getNextWeekDate()))
  }

  "a DateMatcher" should "be catching -1d and +2w unformatted german dates" taggedAs FastTest in {

    val data: Dataset[Row] = DataBuilder.basicDataBuild(
      "Ich habe ihn gestern getroffen und er hat mir gesagt, dass er uns in 2 wochen besuchen wird.")

    val dateMatcher = new MultiDateMatcher()
      .setInputCols("document")
      .setOutputCol("date")
      .setFormat("MM/dd/yyyy")
      .setSourceLanguage("de")

    val pipeline = new Pipeline().setStages(Array(dateMatcher))

    val annotated = pipeline.fit(data).transform(data)

    val annotations: Seq[Annotation] =
      Annotation.getAnnotations(
        annotated.select("date").collect().head,
        "date")

    val results: Set[String] = annotations.map(_.result).toSet

    assert(results.contains(getOneDayAgoDate()) && results.contains(getInTwoWeeksDate()))
  }

}
