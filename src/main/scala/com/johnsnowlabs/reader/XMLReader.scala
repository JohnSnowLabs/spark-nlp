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

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.xml.{Elem, Node, XML}

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
    val xml = XML.loadString(xmlString)
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

          // Traverse children
          elem.child.foreach(traverse(_, Some(elementId)))

        case _ => // Ignore other types
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
