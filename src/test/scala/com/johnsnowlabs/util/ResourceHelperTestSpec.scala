package com.johnsnowlabs.util

import java.io.{File, FileNotFoundException}

import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.scalatest.FlatSpec


class ResourceHelperTestSpec extends FlatSpec {
  // Access spark to avoid creating in ResourceHelper
  private val spark = SparkAccessor.spark

  "List directory" should "correctly list file inside resource directory" in {
    val files = ResourceHelper.listResourceDirectory("ner-dl").toList
    val targetFiles = new File("src/main/resources/ner-dl").list.map{
      f => "ner-dl" + File.separator + f
    }.toList
    assert(files.sorted == targetFiles.sorted)
  }

  "List directory" should "throw file not exists if there is no such file" in {
    assertThrows[FileNotFoundException](ResourceHelper.listResourceDirectory("not-exists"))
  }

  "Resource helper" should "transform files' content in an array of string representation" in {

    val externalResource = ExternalResource("src/test/resources/resource-helper", ReadAs.LINE_BY_LINE,
                                            Map.empty[String, String])
    val stringRepresentation = ResourceHelper.getFilesContentAsArray(externalResource)
    val expectedStringRepresentation = Array("Hello\nWorld", "Bye\nWorld")

    assert(expectedStringRepresentation.toList == stringRepresentation.toList)

  }

  it should "raise an error when SPARK_DATASET is set in RedAs parameter" in {

    val externalResource = ExternalResource("src/test/resources/resource-helper", ReadAs.SPARK_DATASET,
      Map("format"->"text"))
    val caught = intercept[Exception] {
      ResourceHelper.getFilesContentAsArray(externalResource)
    }

    assert(caught.getMessage == "Unsupported readAs")
  }

  it should "raise FileNotFound exception when a wrong path is sent" in {

    val externalResource = ExternalResource("wrong/path/", ReadAs.LINE_BY_LINE,
      Map.empty[String, String])
    val expectedMessage = "folder: wrong/path/ not found"

    assertThrows[FileNotFoundException]{
      ResourceHelper.getFilesContentAsArray(externalResource)
    }

    val caught = intercept[FileNotFoundException] {
      ResourceHelper.getFilesContentAsArray(externalResource)
    }

    assert(caught.getMessage == expectedMessage)

  }

  "Resource helper" should "valid a file" in {
    val rightFilePath = "src/test/resources/parser/dependency_treebank/wsj_0001.dp"
    val isValid = ResourceHelper.validFile(rightFilePath)

    assert(isValid)

  }

  it should "also valid a directory" in {
    val rightDirectoryPath = "src/test/resources/parser/dependency_treebank"
    val isValid = ResourceHelper.validFile(rightDirectoryPath)

    assert(isValid)
  }

  it should "raise FileNotFound exception when an invalid file name is used" in {
    val rightFilePath = "src/test/resources/parser/dependency_treebank/invalid_file_name.dp"

    assertThrows[FileNotFoundException]{
      ResourceHelper.validFile(rightFilePath)
    }

  }

  it should "raise FileNotFound exception when an invalid directory path is used" in {
    val rightFilePath = "wrong/path//wsj_0001.dp"

    assertThrows[FileNotFoundException]{
      ResourceHelper.validFile(rightFilePath)
    }

  }

}

