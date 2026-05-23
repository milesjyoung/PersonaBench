# Adds a deterministic leakage audit to Step 5 task generation.

## New workflow:

### generate test cases
→ run lexical leakage/directness audit
→ write audit report
→ run existing LLM verification
→ accept only if leakage audit AND verification both pass
→ otherwise retry within --max-iterations
→ Note: set metric thresholds at top of leakage_audit.py (controls pass/fail strictness)

## Purpose:

Detect task sets where Type 4 / Type 5 answers are too directly recoverable
because ground_truth labels appear answer-shaped in app logs.

No repair pass yet. Failed leakage attempts currently trigger full regeneration using the existing iteration loop. Existing LLM verification remains unchanged. Step 5 already regenerates when verification fails, so this adds a second deterministic failure condition.

## Future work:

Add targeted repair pass for failed Type 4/5 cases instead of regenerating
the full task set.


## Optional Reading - audit metrics:

### question_leakage_flag_rate
Questions (tasks) that were flagged as having potentially answer-shaped question wording.
It looks at several lexical similarity signals:
* question-ground_truth token overlap
* how many ground-truth tokens appear in the question
* rare/important answer terms appearing in the question
* long shared phrases
* general string similarity

### Directness
These metrics measure the directness of task ground truth contained in the app logs. These values are calculated internally using the following lexical metrics:
* gt_containment: What fraction of meaningful ground-truth tokens appear in this log entry?
* rare_containment: The script treats these as important -- longer words, numbers, money amounts, dates, percentages
* jaccard: This compares token-set overlap
* sequence_ratio: This uses Python’s difflib.SequenceMatcher to check broader string similarity. It catches cases where a log entry is phrased very similarly to the ground truth.
* similarity: The script combines those signals by taking the maximum of a few scores -- sim= max(
    jac,
    0.65 * gt_cont + 0.35 * rare_cont,
    seq,
)

#### avg_evidence_directness_level
This is the average directness score across all tasks.
Each task gets a score from 0 to 4:

| Level | Meaning |
|---|---|
| 0 | No obvious lexical support in logs |
| 1 | Weak / indirect lexical support |
| 2 | Moderate support |
| 3 | Direct or answer-shaped support in one log entry |
| 4 | Direct or answer-shaped support repeated across multiple log entries |

#### pct_directness_3_or_4
This is the percentage of tasks whose directness score is high, meaning level 3 or 4.

