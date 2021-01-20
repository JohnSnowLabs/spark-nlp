package com

import org.scalatest.Tag
import com.johnsnowlabs.build.BuildInfo

package object johnsnowlabs {
  object FastTest extends Tag(s"${BuildInfo.name}.fast")
  object SlowTest extends Tag(s"${BuildInfo.name}.slow")
}