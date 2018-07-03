package com.johnsnowlabs.nlp

import java.io.InputStream

trait HasOcr {

  private[nlp] def doOcr(fileStream:InputStream):Seq[(Int, String)]
  private[nlp] def annotate(path: String, region: String, pageN: Int): Seq[Annotation]

}
