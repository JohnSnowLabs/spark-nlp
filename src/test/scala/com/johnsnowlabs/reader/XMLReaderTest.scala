package com.johnsnowlabs.reader

import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._

class XMLReaderTest extends AnyFlatSpec {

  val xmlFilesDirectory = "./src/test/resources/reader/xml/"

  "XMLReader" should "run end to end pipeline test" taggedAs SlowTest in {
    val reader = new XMLReader()
    val xmlDF = reader.read(s"$xmlFilesDirectory/test.xml")

    val flatDf = xmlDF
      .withColumn("text", expr("concat_ws('\n', xml.content)"))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("token")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer))

    val result = pipeline.fit(flatDf).transform(flatDf)
    result.show(false)

    assert(result.columns.contains("document"))
    assert(result.columns.contains("token"))
    assert(result.select("document").count() > 0)
    assert(result.select("token").count() > 0)
    assert(!result.select("token.result").isEmpty)

    val tokens = result.select("token.result").as[Seq[String]].head
    assert(tokens.nonEmpty)
  }

  "XMLReader" should "read xml as dataframe" taggedAs FastTest in {
    val XMLReader = new XMLReader()
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/test.xml")
    xmlDF.show(truncate = false)

    assert(!xmlDF.select(col("xml").getItem(0)).isEmpty)
    assert(!xmlDF.columns.contains("content"))

    val expectedText =
      """Harry Potter
        |J K. Rowling
        |2005
        |29.99
        |Learning XML
        |Erik T. Ray
        |2003
        |39.95""".stripMargin

    import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

    val collected = xmlDF.select("xml.content").as[Array[String]].collect()

    val text: String = collected.head.mkString("\n")
    assert(text == expectedText)
  }

  // FIXME: What kind of tag behavior do we want?
  it should "include tags in the output" taggedAs FastTest ignore {
    val XMLReader = new XMLReader(xmlKeepTags = true)
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/multi-level.xml")
    xmlDF.show(truncate = false)

    val explodedDf = xmlDF.withColumn("xml_exploded", explode(col("xml")))
    val tagsDf = explodedDf.filter(col("xml_exploded.metadata")("tag") =!= "")

    assert(tagsDf.count() > 0)
  }

  // FIXME: Do we really want to include empty nodes in the output? Since we now have extract attributes option
  it should "output all nodes" taggedAs FastTest ignore {
    val XMLReader = new XMLReader(onlyLeafNodes = false)
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/multi-level.xml")
    xmlDF.show(truncate = false)
    val explodedDf = xmlDF.withColumn("xml_exploded", explode(col("xml")))

    val noParentIdCount = explodedDf
      .filter(!array_contains(map_keys(col("xml_exploded.metadata")), "parentId"))

    assert(noParentIdCount.count() > 0)
  }

  // FIXME: I guess we don't need this anymore
  it should "extract attributes as NARRATIVE_TEXT elements" taggedAs FastTest ignore {
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

  // FIXME: Not relevant anymore?
  it should "link attribute elements to their parentId" taggedAs FastTest ignore {
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

  "XMLReader" should "extract attributes as text" taggedAs FastTest in {
    val XMLReader = new XMLReader(extractTagAttributes = Set("category", "lang"))
    val xmlDF = XMLReader.read(s"$xmlFilesDirectory/test.xml")
    xmlDF.show(truncate = false)

    assert(!xmlDF.select(col("xml").getItem(0)).isEmpty)
    assert(!xmlDF.columns.contains("content"))

    val expectedText =
      """children
        |en
        |Harry Potter
        |J K. Rowling
        |2005
        |29.99
        |web
        |en
        |Learning XML
        |Erik T. Ray
        |2003
        |39.95""".stripMargin

    import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._

    val collected = xmlDF.select("xml.content").as[Array[String]].collect()

    val text: String = collected.head.mkString("\n")
    assert(text == expectedText)
  }
}
