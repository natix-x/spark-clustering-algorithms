package clustering.algorithms.dbscan

import org.log4s.getLogger

/** Time-throttled progress over the m²/2 pair space (counts pairs, not rows — row i scans
 *  m − i others, so a row percentage would run far ahead of the truth). */
private[dbscan] final class ScanProgress(totalRows: Int) {
  private val logger = getLogger

  private val totalPairs = totalRows.toDouble * totalRows / 2
  private val startedAt = System.nanoTime()
  private var lastLogAt = startedAt

  private val LOG_INTERVAL_NANOS = 30L * 1000000000L

  def report(rowsDone: Int, path: String): Unit = {
    val now = System.nanoTime()
    if (now - lastLogAt >= LOG_INTERVAL_NANOS) {
      lastLogAt = now
      val elapsed = (now - startedAt) / 1e9
      val done = (rowsDone.toDouble * totalRows - rowsDone.toDouble * rowsDone / 2) / totalPairs
      logger.info(f"dbscanpp: step 3 ε-graph ($path) ${done * 100}%.1f%% " +
        f"($rowsDone%d/$totalRows%d rows), $elapsed%.0f s elapsed, " +
        f"${elapsed * (1 - done) / done}%.0f s left")
    }
  }
}
