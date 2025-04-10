package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.util.io.ResourceHelper.validFile
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

  private val spark = ResourceHelper.spark
  import spark.implicits._

  def read(inputSource: String): DataFrame = {
    if (validFile(inputSource)) {
      val xmlDf = spark.sparkContext
        .wholeTextFiles(inputSource)
        .toDF("path", "content")
        .withColumn("xml", parseHtmlUDF(col("content")))
      if (storeContent) xmlDf.select("path", "content", "xml")
      else xmlDf.select("path", "xml")
    } else throw new IllegalArgumentException(s"Invalid inputSource: $inputSource")
  }

  private val parseHtmlUDF = udf((html: String) => {
    parseXml(html)
  })

  private def parseXml(xmlString: String): List[HTMLElement] = {
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
