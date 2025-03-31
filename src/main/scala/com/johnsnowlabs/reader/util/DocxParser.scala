/*
 * Copyright 2017-2024 John Snow Labs
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
package com.johnsnowlabs.reader.util

import org.apache.poi.xwpf.usermodel.{
  ParagraphAlignment,
  XWPFDocument,
  XWPFParagraph,
  XWPFRun,
  XWPFTable
}

import scala.collection.JavaConverters._

object DocxParser {

  implicit class RichXWPFParagraph(paragraph: XWPFParagraph) {

    private def isListParagraph: Boolean =
      paragraph.getStyle != null && paragraph.getStyle.startsWith("List")

    private def isNumberList: Boolean = {
      val numId = paragraph.getNumID
      numId != null && paragraph.getDocument.getNumbering.getNum(numId) != null
    }

    private def isBulletPoint: Boolean = {
      if (paragraph.getNumID == null) {
        // Check for common bullet characters, indentation, or specific styling
        val text = paragraph.getText.trim
        text.startsWith("•") || text.startsWith("–") || text.startsWith("*") ||
        paragraph.getIndentationLeft > 0 ||
        paragraph.getRuns.toArray.exists { case run: XWPFRun =>
          run.getFontFamily == "Wingdings"
        }
      } else {
        false
      }
    }

    def isListItem: Boolean = {
      isListParagraph || isNumberList || isBulletPoint
    }

    def isTitle: Boolean = {
      (paragraph.getStyle != null && paragraph.getStyle.startsWith("Heading")) ||
      isBold && isCentered ||
      isBold && isUppercaseOrCapitalized
    }

    def isBold: Boolean = paragraph.getRuns.toArray.exists { case run: XWPFRun =>
      run.isBold
    }

    def isCentered: Boolean = {
      val alignment = paragraph.getAlignment
      alignment == ParagraphAlignment.CENTER || alignment == ParagraphAlignment.BOTH
    }

    private def isUppercaseOrCapitalized: Boolean = {
      val text = paragraph.getText.trim
      text.nonEmpty && (text.forall(_.isUpper) ||
        text.split("\\s+").exists(word => word.headOption.exists(_.isUpper)))
    }

    def isCustomPageBreak: Boolean = {
      val ctp = paragraph.getCTP // Get the paragraph's XML representation
      Option(ctp.getDomNode).exists { node =>
        val allNodes = getAllNodes(node)
        allNodes.exists { child =>
          // Check for manual page break
          (child.getNodeName == "w:br" &&
            Option(child.getAttributes).exists(attrs =>
              Option(attrs.getNamedItem("w:type")).exists(_.getNodeValue == "page"))) ||
          // Check for rendered page break
          child.getNodeName == "w:lastRenderedPageBreak"
        }
      }
    }

    // Helper function to traverse all child nodes recursively
    private def getAllNodes(node: org.w3c.dom.Node): Seq[org.w3c.dom.Node] = {
      val children = node.getChildNodes
      (0 until children.getLength).flatMap { i =>
        val child = children.item(i)
        Seq(child) ++ getAllNodes(child)
      }
    }

    def isSectionBreak: Boolean = {
      val ctp = paragraph.getCTP
      Option(ctp.getPPr).exists(_.isSetSectPr)
    }

  }

  implicit class RichXWPFDocument(document: XWPFDocument) {

    def extractHeaders: Seq[String] = {
      val headerFooterPolicy = Option(document.getHeaderFooterPolicy)
      headerFooterPolicy.toSeq.flatMap { policy =>
        Seq(
          Option(policy.getDefaultHeader),
          Option(policy.getFirstPageHeader),
          Option(policy.getEvenPageHeader)).flatten
          .flatMap { header =>
            header.getParagraphs.asScala.map { paragraph =>
              paragraph.getText.trim
            }
          }
          .filter(_.nonEmpty)
      }
    }

    def extractFooters: Seq[String] = {
      val headerFooterPolicy = Option(document.getHeaderFooterPolicy)
      headerFooterPolicy.toSeq.flatMap { policy =>
        Seq(
          Option(policy.getDefaultFooter),
          Option(policy.getFirstPageFooter),
          Option(policy.getEvenPageFooter)).flatten
          .flatMap { footer =>
            footer.getParagraphs.asScala.map { paragraph =>
              paragraph.getText.trim
            }
          }
          .filter(_.nonEmpty)
      }
    }
  }

  implicit class RichXWPFTable(table: XWPFTable) {

    def processAsHtml: String = {
      val htmlRows = table.getRows.asScala.zipWithIndex
        .map { case (row, rowIndex) =>
          val cellsHtml = row.getTableCells.asScala
            .map { cell =>
              val cellText = cell.getText
              if (rowIndex == 0) s"<th>$cellText</th>" else s"<td>$cellText</td>"
            }
            .mkString("")
          s"<tr>$cellsHtml</tr>"
        }
        .mkString("")
      s"<table>$htmlRows</table>"
    }

  }

}
