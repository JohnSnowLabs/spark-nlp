package com.johnsnowlabs.util

import java.io.{File, FileNotFoundException}
import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest.FlatSpec


class ResourceHelperSpec extends FlatSpec {
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
}

