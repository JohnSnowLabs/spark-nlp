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

import com.amazonaws.AmazonServiceException
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.scalatest.flatspec.AnyFlatSpec

import java.io.{File, FileNotFoundException}
import java.nio.file.Paths

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
      ResourceHelper.SourceStream("src/test/resources/entity-ruler/keywords_with_id.jsonl")
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

  it should "parse S3 URIs" taggedAs FastTest in {
    val s3URIs =
      Array("s3a://my.bucket.com/my/S3/path/my_file.tmp", "s3://my.bucket.com/my/S3/path/")
    val expectedOutput =
      Array(("my.bucket.com", "my/S3/path/my_file.tmp"), ("my.bucket.com", "my/S3/path/"))

    s3URIs.zipWithIndex.foreach { case (s3URI, index) =>
      val (actualBucket, actualKey) = ResourceHelper.parseS3URI(s3URI)

      val (expectedBucket, expectedKey) = expectedOutput(index)

      assert(expectedBucket == actualBucket)
      assert(expectedKey == actualKey)
    }
  }

  it should "not copyToLocal a local file" taggedAs FastTest in {
    val resourcePath = "src/test/resources/tf-hub-bert/model"
    val resourceUri = new File("src/test/resources/tf-hub-bert/model").getAbsolutePath

    val tmpFolder = ResourceHelper.copyToLocal(resourcePath)

    assert(resourceUri == tmpFolder, "Folder should not have been copied.")
  }

  // Local HDFS needs to be set up
  ignore should "copyToLocal from hdfs" taggedAs SlowTest in {

    // Folder
    val hdfsFolderPath = "hdfs://localhost:9000/sparknlp/tf-hub-bert/model"
    val resourcePath = "src/test/resources/tf-hub-bert/model"
    val resourceFolderContent: Array[String] = new File(resourcePath).listFiles().map(_.getName)
    val tmpFolder = ResourceHelper.copyToLocal(hdfsFolderPath)

    val localPath = new File(tmpFolder)

    localPath.listFiles().foreach { f: File =>
      assert(
        resourceFolderContent.contains(f.getName),
        s"File $f missing in copied temporary folder $tmpFolder.")
    }

    // Single File
    val hdfsFilePath = "hdfs://localhost:9000/sparknlp/tf-hub-bert/model/assets/vocab.txt"

    val tmpFolderFile: String = ResourceHelper.copyToLocal(hdfsFilePath)
    assert(Paths.get(tmpFolderFile, "vocab.txt").toFile.exists(), "Copied file doesn't exist.")
  }

  // AWS keys need to be set up for this test
  ignore should "copyToLocal from s3" taggedAs SlowTest in {
    val awsAccessKeyId = sys.env("AWS_ACCESS_KEY_ID")
    val awsSecretAccessKey = sys.env("AWS_SECRET_ACCESS_KEY")
    val awsSessionToken = sys.env("AWS_SESSION_TOKEN")

    ResourceHelper.getSparkSessionWithS3(
      awsAccessKeyId,
      awsSecretAccessKey,
      awsSessionToken = Some(awsSessionToken))

    val s3FolderPath = "s3://sparknlp-test/tf-hub-bert/model"

    val resourcePath = "src/test/resources/tf-hub-bert/model"

    val resourceFolderContent: Array[String] = new File(resourcePath).listFiles().map(_.getName)

    val localPath = ResourceHelper.copyToLocal(s3FolderPath)
    new File(localPath).listFiles().foreach { f: File =>
      assert(
        resourceFolderContent.contains(f.getName),
        s"File $f missing in copied temporary folder $localPath.")
    }

  }

  ignore should "copyToLocal should catch s3 exception" taggedAs SlowTest in {
    ResourceHelper.getSparkSessionWithS3("NONE", "NONE", awsSessionToken = Some("NONE"))

    val s3FolderPath = "s3://sparknlp-test/tf-hub-bert/model"

    assertThrows[AmazonServiceException] {
      ResourceHelper.copyToLocal(s3FolderPath)
    }
  }

}
