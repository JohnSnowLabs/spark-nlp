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

package com.johnsnowlabs.util

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{File, FileNotFoundException}

class ResourceHelperTestSpec extends AnyFlatSpec {

  "Resource helper" should "load a file line by line as an array" taggedAs FastTest in {
    val externalResource = ExternalResource(
      "src/test/resources/resource-helper/gender.tsv",
      ReadAs.TEXT,
      Map("delimiter" -> "\t"))
    val expectedDictionary =
      Map("Female" -> List("lady", "women"), "Male" -> List("man", "boy", "male", "son"))

    val dictionary = ResourceHelper.parseKeyListValues(externalResource)

    assert(dictionary == expectedDictionary)
  }

  "List directory" should "correctly list file inside resource directory" taggedAs FastTest in {
    val files = ResourceHelper.listResourceDirectory("ner-dl").toList
    val targetFiles = new File("src/main/resources/ner-dl").list.map { f =>
      "ner-dl" + File.separator + f
    }.toList
    assert(files.sorted == targetFiles.sorted)
  }

  "List directory" should "throw file not exists if there is no such file" taggedAs FastTest in {
    assertThrows[FileNotFoundException](ResourceHelper.listResourceDirectory("not-exists"))
  }

  "Resource helper" should "transform files' content in an array of string representation" taggedAs FastTest in {

    val externalResource = ExternalResource(
      "src/test/resources/resource-helper",
      ReadAs.TEXT,
      Map.empty[String, String])
    val iteratorRepresentation = ResourceHelper.getFilesContentBuffer(externalResource)
    val expectedIteratorRepresentation =
      Seq(Array(s"ByeWorld").toIterator, Array(s"HelloWorld").toIterator)

    val stringRepresentation = iteratorRepresentation.map(line => line.mkString)
    val expectedStringRepresentation = expectedIteratorRepresentation.map(line => line.mkString)

    assert(expectedStringRepresentation.forall(stringRepresentation contains _))

  }

  it should "raise an error when SPARK is set in RedAs parameter" taggedAs FastTest in {

    val externalResource = ExternalResource(
      "src/test/resources/resource-helper",
      ReadAs.SPARK,
      Map("format" -> "text"))
    val caught = intercept[Exception] {
      ResourceHelper.getFilesContentBuffer(externalResource)
    }

    assert(caught.getMessage == "Unsupported readAs")
  }

  it should "raise FileNotFound exception when a wrong path is sent" taggedAs FastTest in {

    val externalResource = ExternalResource("wrong/path/", ReadAs.TEXT, Map.empty[String, String])
    val expectedMessage = "file or folder: wrong/path/ not found"

    assertThrows[FileNotFoundException] {
      ResourceHelper.getFilesContentBuffer(externalResource)
    }

    val caught = intercept[FileNotFoundException] {
      ResourceHelper.getFilesContentBuffer(externalResource)
    }

    assert(caught.getMessage == expectedMessage)

  }

  "Resource helper" should "valid a file" taggedAs FastTest in {
    val rightFilePath = "src/test/resources/parser/unlabeled/dependency_treebank/wsj_0001.dp"
    val isValid = ResourceHelper.validFile(rightFilePath)

    assert(isValid)

  }

  it should "also valid a directory" taggedAs FastTest in {
    val rightDirectoryPath = "src/test/resources/parser/unlabeled/dependency_treebank"
    val isValid = ResourceHelper.validFile(rightDirectoryPath)

    assert(isValid)
  }

  it should "return false when an invalid file name is used" taggedAs FastTest in {
    val rightFilePath =
      "src/test/resources/parser/unlabeled/dependency_treebank/invalid_file_name.dp"

    val isValid = ResourceHelper.validFile(rightFilePath)

    assert(!isValid)
  }

  it should "raise FileNotFound exception when an invalid directory path is used" taggedAs FastTest in {
    val rightFilePath = "wrong/path//wsj_0001.dp"

    val isValid = ResourceHelper.validFile(rightFilePath)

    assert(!isValid)

  }

  it should "get content from SourceStream" taggedAs FastTest in {
    val sourceStream =
      ResourceHelper.SourceStream("src/test/resources/entity-ruler/patterns.jsonl")
    val expectedContent = Array(
      "{\"id\": \"names-with-j\", \"label\": \"PERSON\", \"patterns\": [\"Jon\", \"John\", \"John Snow\", \"Jon Snow\"]}",
      "{\"id\": \"names-with-s\", \"label\": \"PERSON\", \"patterns\": [\"Stark\"]}",
      "{\"id\": \"names-with-e\", \"label\": \"PERSON\", \"patterns\": [\"Eddard\", \"Eddard Stark\"]}",
      "{\"id\": \"locations\", \"label\": \"LOCATION\", \"patterns\": [\"Winterfell\"]}")
    var actualContent: Array[String] = Array()

    sourceStream.content.foreach(content =>
      content.foreach(c => actualContent = actualContent ++ Array(c)))

    assert(expectedContent sameElements actualContent)
  }

  it should "list files" in {
    val files = ResourceHelper.listLocalFiles("src/test/resources/image")

    assert(files.nonEmpty)
  }

}
