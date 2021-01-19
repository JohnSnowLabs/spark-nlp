package com

import org.scalatest.Tag

package object johnsnowlabs {
  object FastTest extends Tag("com.johnsnowlabs.test.tags.fast")
  object SlowTest extends Tag("com.johnsnowlabs.test.tags.slow")
}
