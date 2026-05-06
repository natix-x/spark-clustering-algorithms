name := "spark-clustering"
version := "0.1"

scalaVersion := "2.12.15"

val sparkVersion = "3.3.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"   % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql"    % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib"  % sparkVersion % "provided",
  "org.apache.commons" % "commons-math3" % "3.6.1"
)

//assembly / assemblyMergeStrategy := {
//  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
//  case x if x.endsWith("module-info.class") => MergeStrategy.discard
//  case x => MergeStrategy.first
//}

exportJars := true