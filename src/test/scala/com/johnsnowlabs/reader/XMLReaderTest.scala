package com.johnsnowlabs.reader

import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.functions.{array_contains, col, explode, map_keys}
import org.scalatest.flatspec.AnyFlatSpec

class XMLReaderTest extends AnyFlatSpec {

  val xmlFilesDirectory = "./src/test/resources/reader/xml/"

  "XMLReader" should "read xml as dataframe" taggedAs FastTest in {
    val XMLReader = new XMLReader()
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/test.xml")
    xmlDF.show(truncate = false)

    assert(!xmlDF.select(col("xml").getItem(0)).isEmpty)
    assert(!xmlDF.columns.contains("content"))
  }

  it should "include tags in the output" taggedAs FastTest in {
    val XMLReader = new XMLReader(xmlKeepTags = true)
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/multi-level.xml")
    xmlDF.show(truncate = false)

    val explodedDf = xmlDF.withColumn("xml_exploded", explode(col("xml")))
    val tagsDf = explodedDf.filter(col("xml_exploded.metadata")("tag") =!= "")

    assert(tagsDf.count() > 0)
  }

  it should "output all nodes" taggedAs FastTest in {
    val XMLReader = new XMLReader(onlyLeafNodes = false)
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/multi-level.xml")
    xmlDF.show(truncate = false)
    val explodedDf = xmlDF.withColumn("xml_exploded", explode(col("xml")))

    val noParentIdCount = explodedDf
      .filter(!array_contains(map_keys(col("xml_exploded.metadata")), "parentId"))

    assert(noParentIdCount.count() > 0)
  }

  it should "extract attributes as NARRATIVE_TEXT elements" taggedAs FastTest in {
    val xml =
      """<root>
        |  <observation code="ASSERTION" statusCode="completed"/>
        |</root>""".stripMargin

    val reader = new XMLReader(xmlKeepTags = true, onlyLeafNodes = true)
    val elements = reader.parseXml(xml)

    val attrElements = elements.filter(_.elementType == ElementType.NARRATIVE_TEXT)

    assert(attrElements.nonEmpty, "Attributes should be extracted as NARRATIVE_TEXT")

    val codeAttrOpt =
      attrElements.find(_.metadata.get("attribute").exists(_.equalsIgnoreCase("code")))
    assert(codeAttrOpt.isDefined, "Expected attribute 'code' was not found")
    assert(codeAttrOpt.get.content == "ASSERTION")

    val statusAttrOpt =
      attrElements.find(_.metadata.get("attribute").exists(_.equalsIgnoreCase("statusCode")))
    assert(statusAttrOpt.isDefined, "Expected attribute 'statusCode' was not found")
    assert(statusAttrOpt.get.content == "completed")
  }

  it should "link attribute elements to their parentId" taggedAs FastTest in {
    val xml =
      """<root>
        |  <item id="123" class="test">Content</item>
        |</root>""".stripMargin

    val reader = new XMLReader(xmlKeepTags = true, onlyLeafNodes = true)
    val elements = reader.parseXml(xml)

    val itemElem = elements.find(e => e.metadata.get("tag").contains("item")).get
    val attrElems = elements.filter(_.metadata.contains("attribute"))

    assert(attrElems.forall(_.metadata("parentId") == itemElem.metadata("elementId")))
  }

}
