resolvers += "Maven Central" at "https://repo1.maven.org/maven2/"

addSbtPlugin("org.xerial.sbt" % "sbt-sonatype" % "3.11.2")
addSbtPlugin("com.github.sbt" % "sbt-pgp" % "2.1.2")

addSbtPlugin("org.scalameta" % "sbt-scalafmt" % "2.4.6")

/** scoverage */
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "1.9.3")
addSbtPlugin("org.scoverage" % "sbt-coveralls" % "1.3.2")

addDependencyTreePlugin
