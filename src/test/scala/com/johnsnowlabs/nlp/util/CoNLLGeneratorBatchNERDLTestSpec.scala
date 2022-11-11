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

package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

class CoNLLGeneratorBatchNERDLTestSpec extends AnyFlatSpec {
  ResourceHelper.spark // for toDS and toDF

  def formatCoNLL(
      firstname: String,
      lastname: String,
      verb: String,
      proposition: String,
      company: String,
      country: String): String = {
    s"""
       |$firstname PER
       |$lastname PER
       |$verb O
       |$proposition O
       |$company ORG
       |$country LOC
       |. O
       |""".stripMargin
  }

  "The (dataframe, pipelinemodel, outputpath) generator" should "make the right CoNLL file" taggedAs SlowTest in {

    val firstnames = Array(
      "Liam",
      "Olivia",
      "Noah",
      "Emma",
      "Oliver",
      "Ava",
      "William",
      "Sophia",
      "Elijah",
      "Isabella")
    val lastnames = Array(
      "Smith",
      "Johnson",
      "Williams",
      "Brown",
      "Jones",
      "Garcia",
      "Miller",
      "Davis",
      "Rodriguez",
      "Martinez")
    val verbs = Array("be", "have", "do", "say", "go", "get", "make", "known", "think", "take")
    val propositions = Array("of", "in", "to", "for", "with", "on", "at", "from", "by", "about")
    val company = Array(
      "Walmart",
      "Amazon.com",
      "PetroChina",
      "Apple",
      "CVS",
      "RoyalDutch",
      "Berkshire",
      "Google",
      "FaceBook",
      "Fiat")
    val country = Array(
      "China",
      "India",
      "USA",
      "Indonesia",
      "Pakistan",
      "Brazil",
      "Nigeria",
      "Bangladesh",
      "Russia",
      "Mexico")

    // Modify simulation index i for more data volume
    for (i <- 1 to 6) {
      val nerTagsDatasetStr: Array[String] = {
        for (firstname <- firstnames.take(i);
          lastname <- lastnames.take(i);
          verb <- verbs.take(i);
          proposition <- propositions.take(i);
          company <- company.take(i);
          country <- country.take(i))
          yield formatCoNLL(firstname, lastname, verb, proposition, company, country)
      }

      val prefix = "-DOCSTART- O\n"
      val suffix = "\n"

      Files.write(
        Paths.get(s"./tmp_ner_fake_conll_$i.txt"),
        (prefix + nerTagsDatasetStr.mkString + suffix)
          .getBytes(StandardCharsets.UTF_8))
    }
  }
}
