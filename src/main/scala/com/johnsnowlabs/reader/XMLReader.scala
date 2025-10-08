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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.validFile
import com.johnsnowlabs.partition.util.PartitionHelper.datasetWithTextFile
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}
import org.ccil.cowan.tagsoup.jaxp.SAXFactoryImpl
import org.xml.sax.InputSource

import java.io.StringReader
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.xml.parsing.NoBindingFactoryAdapter
import scala.xml.{Elem, Node, XML}

/** Class to parse and read XML files.
  *
  * @param storeContent
  *   Whether to include the raw XML content in the resulting DataFrame as a separate 'content'
  *   column. By default, this is false.
  *
  * @param xmlKeepTags
  *   Whether to retain original XML tag names and include them in the metadata for each extracted
  *   element. Useful for preserving structure. Default is false.
  *
  * @param onlyLeafNodes
  *   If true, only the deepest elements (those without child elements) are extracted. If false,
  *   all elements are extracted. Default is true.
  *
  * ==Input Format==
  * Input must be a valid path to an XML file or a directory containing XML files.
  *
  * ==Example==
  * {{{
  * val xmlPath = "./data/sample.xml"
  * val xmlReader = new XMLReader()
  * val xmlDf = xmlReader.read(xmlPath)
  * }}}
  *
  * {{{
  * xmlDf.show(truncate = false)
  * +----------------------+--------------------------------------------------+
  * |path                  |xml                                               |
  * +----------------------+--------------------------------------------------+
  * |file:/data/sample.xml |[{Title, My Book, {tag -> title}}, ...]          |
  * +----------------------+--------------------------------------------------+
  *
  * xmlDf.printSchema()
  * root
  *  |-- path: string (nullable = true)
  *  |-- xml: array (nullable = true)
  *  |    |-- element: struct (containsNull = true)
  *  |    |    |-- elementType: string (nullable = true)
  *  |    |    |-- content: string (nullable = true)
  *  |    |    |-- metadata: map (nullable = true)
  *  |    |    |    |-- key: string
  *  |    |    |    |-- value: string (valueContainsNull = true)
  * }}}
  *
  * For more examples refer to:
  * [[https://github.com/JohnSnowLabs/spark-nlp/examples/python/reader/SparkNLP_XML_Reader_Demo.ipynb notebook]]
  */
class XMLReader(
    storeContent: Boolean = false,
    xmlKeepTags: Boolean = false,
    onlyLeafNodes: Boolean = true)
    extends Serializable {

  private lazy val spark = ResourceHelper.spark

  private var outputColumn = "xml"

  def setOutputColumn(value: String): this.type = {
    require(value.nonEmpty, "Output column name cannot be empty.")
    outputColumn = value
    this
  }

  def read(inputSource: String): DataFrame = {
    if (validFile(inputSource)) {
      val xmlDf = datasetWithTextFile(spark, inputSource)
        .withColumn(outputColumn, parseXmlUDF(col("content")))
      if (storeContent) xmlDf.select("path", "content", outputColumn)
      else xmlDf.select("path", outputColumn)
    } else throw new IllegalArgumentException(s"Invalid inputSource: $inputSource")
  }

  private val parseXmlUDF = udf((xml: String) => {
    parseXml(xml)
  })

  def parseXml(xmlString: String): List[HTMLElement] = {
    val parser = new SAXFactoryImpl().newSAXParser()
    val adapter = new NoBindingFactoryAdapter
    val xml = adapter.loadXML(new InputSource(new StringReader(xmlString)), parser)
    val elements = ListBuffer[HTMLElement]()

    def traverse(node: Node, parentId: Option[String]): Unit = {
      node match {
        case elem: Elem =>
          val tagName = elem.label.toLowerCase
          val textContent = elem.text.trim
          val elementId = hash(tagName + textContent)

          val isLeaf = !elem.child.exists(_.isInstanceOf[Elem])

          if (!onlyLeafNodes || isLeaf) {
            val elementType = tagName match {
              case "title" | "author" => ElementType.TITLE
              case _ => ElementType.UNCATEGORIZED_TEXT
            }

            val metadata = mutable.Map[String, String]("elementId" -> elementId)
            if (xmlKeepTags) metadata += ("tag" -> tagName)
            parentId.foreach(id => metadata += ("parentId" -> id))

            val content = if (isLeaf) textContent else ""
            elements += HTMLElement(elementType, content, metadata)
          }

          elem.attributes.asAttrMap.foreach { case (attrName, attrValue) =>
            val attrId = hash(tagName + attrName + attrValue)
            val metadata =
              mutable.Map("elementId" -> attrId, "parentId" -> elementId, "attribute" -> attrName)
            if (xmlKeepTags) metadata += ("tag" -> tagName)

            elements += HTMLElement(ElementType.NARRATIVE_TEXT, attrValue, metadata)
          }

          // Traverse children
          elem.child.foreach(traverse(_, Some(elementId)))

        case _ => // ignore
      }
    }

    traverse(xml, None)
    elements.toList
  }

  def hash(s: String): String = {
    java.security.MessageDigest
      .getInstance("MD5")
      .digest(s.getBytes)
      .map("%02x".format(_))
      .mkString
  }

}
