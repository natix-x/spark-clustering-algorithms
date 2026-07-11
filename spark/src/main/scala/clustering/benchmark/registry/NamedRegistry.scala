package clustering.benchmark.registry

/** Case-insensitive name -> value lookup with a uniform "unknown X" error.
 *  Shared by the algorithm, data-source and distance registries so all three
 *  resolve names and report unknown ones exactly the same way. */
private[registry] final class NamedRegistry[A](kind: String, entries: Map[String, A]) {
  private val byName: Map[String, A] = entries.map { case (k, v) => k.toLowerCase -> v }

  /** Registered names, sorted — for error messages and discovery. */
  def knownNames: Seq[String] = byName.keys.toSeq.sorted

  def get(name: String): A =
    byName.getOrElse(name.toLowerCase, throw new IllegalArgumentException(
      s"Unknown $kind: '$name'. Known: ${knownNames.mkString(", ")}"))
}

private[registry] object NamedRegistry {
  def apply[A](kind: String, entries: Seq[(String, A)]): NamedRegistry[A] =
    new NamedRegistry(kind, entries.toMap)
}
