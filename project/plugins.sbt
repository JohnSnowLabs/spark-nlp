
resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.foundweekends" % "sbt-bintray" % "0.5.1")

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.9.2")

/** scoverage */
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.6.0")