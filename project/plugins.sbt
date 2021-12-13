resolvers += "bintray-spark-packages" at "https://dl.bintray.com/spark-packages/maven/"

addSbtPlugin("org.foundweekends" % "sbt-bintray" % "0.6.1")

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.9.5")

addSbtPlugin("com.jsuereth" % "sbt-pgp" % "2.0.1")

/** scoverage */
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.8.2")

addSbtPlugin("org.scoverage" % "sbt-coveralls" % "1.3.1")