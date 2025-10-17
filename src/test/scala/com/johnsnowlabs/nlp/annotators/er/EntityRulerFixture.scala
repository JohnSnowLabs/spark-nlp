/*
 * Copyright 2017-2025 John Snow Labs
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
package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.CHUNK

object EntityRulerFixture {

  val email1 = "admin@company.org"
  val email2 = "john.smith@mailserver.net"
  val ip1 = "192.168.1.10"
  val ip2 = "10.0.0.1"
  val text1: String = "John Snow lives in Winterfell"
  val expectedEntitiesFromText1: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(CHUNK, 0, 8, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 19, 28, "Winterfell", Map("entity" -> "LOCATION", "sentence" -> "0"))))
  val expectedEntitiesWithIdFromText1: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        CHUNK,
        0,
        8,
        "John Snow",
        Map("entity" -> "PERSON", "id" -> "names-with-j", "sentence" -> "0")),
      Annotation(
        CHUNK,
        19,
        28,
        "Winterfell",
        Map("entity" -> "LOCATION", "id" -> "locations", "sentence" -> "0"))))

  val text2 = "Lord Eddard Stark was the head of House Stark"
  val expectedEntitiesFromText2: Array[Seq[Annotation]] = Array(
    Seq(Annotation(CHUNK, 5, 16, "Eddard Stark", Map("entity" -> "PERSON", "sentence" -> "0"))))
  val expectedEntitiesWithIdFromText2: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        CHUNK,
        5,
        16,
        "Eddard Stark",
        Map("entity" -> "PERSON", "id" -> "person-regex", "sentence" -> "0"))))

  val text3 =
    "Doctor John Snow lives in London, whereas Lord Commander Jon Snow lives in Castle Black"
  val expectedEntitiesFromText3: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(CHUNK, 0, 15, "Doctor John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 57, 64, "Jon Snow", Map("entity" -> "PERSON", "sentence" -> "0"))))

  val text4 = "In London, John Snow is a Physician. In Castle Black, Jon Snow is a Lord Commander"
  val expectedEntitiesFromText4: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(CHUNK, 11, 19, "John Snow", Map("entity" -> "PERSON", "sentence" -> "0")),
      Annotation(CHUNK, 54, 61, "Jon Snow", Map("entity" -> "PERSON", "sentence" -> "1"))))
  val expectedEntitiesWithIdFromText4: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        CHUNK,
        11,
        19,
        "John Snow",
        Map("entity" -> "PERSON", "id" -> "names-with-j", "sentence" -> "0")),
      Annotation(
        CHUNK,
        54,
        61,
        "Jon Snow",
        Map("entity" -> "PERSON", "id" -> "names-with-j", "sentence" -> "1"))))

  val text5 = "The id of the computer is 192.168.157.3. The id number of this object is 123 456"
  val expectedEntitiesSentenceLevelFromText5: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(CHUNK, 26, 38, "192.168.157.3", Map("entity" -> "ID", "sentence" -> "0")),
      Annotation(CHUNK, 73, 79, "123 456", Map("entity" -> "ID", "sentence" -> "1"))))
  val expectedEntitiesFromText5: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(CHUNK, 26, 38, "192.168.157.3", Map("entity" -> "ID", "sentence" -> "0")),
      Annotation(CHUNK, 73, 75, "123", Map("entity" -> "ID", "sentence" -> "1")),
      Annotation(CHUNK, 77, 79, "456", Map("entity" -> "ID", "sentence" -> "1"))))

  val text6 = "The address is 123456 in Winterfell"
  val expectedEntitiesFromText6: Array[Seq[Annotation]] = Array(
    Seq(
      Annotation(
        CHUNK,
        15,
        20,
        "123456",
        Map("entity" -> "ID", "id" -> "id-regex", "sentence" -> "0")),
      Annotation(
        CHUNK,
        25,
        34,
        "Winterfell",
        Map("entity" -> "LOCATION", "id" -> "locations-words", "sentence" -> "0"))))

}
