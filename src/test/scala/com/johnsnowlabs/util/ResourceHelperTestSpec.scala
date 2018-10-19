package com.johnsnowlabs.util

import java.io.{File, FileNotFoundException}

import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import org.scalatest.FlatSpec


class ResourceHelperTestSpec extends FlatSpec {
  // Access spark to avoid creating in ResourceHelper
  val spark = SparkAccessor.spark

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

  "ResourceHelper" should "transform files' content in an array of string representation" in {

    val externalResource = ExternalResource("/Users/dburbano/tmp/test_files", ReadAs.LINE_BY_LINE,
                                            Map.empty[String, String])
    val stringRepresentation = ResourceHelper.getFilesContentAsArray(externalResource)
    val expectedStringRepresentation = Array("Hello\nWorld", "Bye\nWorld")

    assert(expectedStringRepresentation.toList == stringRepresentation.toList)

  }

  it should "raise an error when SPARK_DATASET is set in RedAs parameter" in {
    val externalResource = ExternalResource("/Users/dburbano/tmp/test_files", ReadAs.SPARK_DATASET,
      Map("format"->"text"))

    val caught = intercept[Exception] {
      ResourceHelper.getFilesContentAsArray(externalResource)
    }
    assert(caught.getMessage == "Unsupported readAs")
  }

}

