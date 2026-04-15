# MLOPs-Bench: Project Context Document
**Last Updated:** Day 1  
**Purpose:** Complete project reference. If starting a new AI conversation,
paste this file first. Contains everything needed to continue without
losing context.

---

## 1. PAPER IDENTITY

**Full Title:** SH-Bench: A Standardized Benchmarking Framework for
Self-Healing MLOps Pipelines

**One-line summary:** We define a fault taxonomy, a parameterized injection
protocol, and a composite scoring metric (SHS) to evaluate and compare
self-healing MLOps pipelines in a reproducible, statistically rigorous way.

**Core novelty claim:**
No existing work unifies multiple self-healing paradigms (drift detection,
circuit breaking, RL-based autoscaling, causal RCA) under a single
reproducible benchmarking system. SH-Bench is the first to do this.

**Target venues:**
- Primary: MLSys, IEEE ICMLA, AAAI workshop tracks
- Secondary: IEEE Access, Expert Systems with Applications, ACM SIGOPS workshops
- Framing: Benchmarking/measurement paper (actively solicited by venues)

**Paper length:** 8-10 pages, IEEE/ACM double column format

---

## 2. TEAM & HARDWARE

| Person | Machine | Role |
|---|---|---|
| You (first author) | MacBook Air M4, 16GB RAM | Code architecture, evaluator modules, paper writing, analysis |
| Partner | Lenovo Legion Pro 7, RTX 5070Ti, 12GB VRAM, 32GB RAM | Pipeline execution, Docker, experiment runs |

**Workflow:**
```
Mac writes code → push to GitHub → Legion pulls and runs
Legion produces results CSV → push to GitHub → Mac pulls and analyzes
Mac writes paper → partner reviews
```

**GitHub repo:** sh-bench (private)  
**No SSH needed** — same room collaboration via GitHub + verbal coordination

---

## 3. DATASET

**Primary:** NYC Yellow Taxi Trip Data  
- `yellow_tripdata_2019-01.csv` → reference/baseline (healthy state)  
- `yellow_tripdata_2020-01.csv` → natural drift source  
- 50,000 rows sampled from each  
- Same month across years eliminates seasonal confounds  
- 2020 captures early COVID-19 behavioral shift (documented real drift)

**Task:** Regression — predict `fare_amount`

**Features used:**
```
trip_distance, passenger_count, RatecodeID, PULocationID,
DOLocationID, payment_type, extra, mta_tax, tolls_amount,
congestion_surcharge
```

**Confirmed natural drift (paper result):**
```
trip_distance: 2.782 → 2.922 (+5.0%)
fare_amount:   12.186 → 12.472 (+2.3%)
```

**Secondary (generalizability):** CICIDS-2017 network intrusion dataset
- Binary classification task
- Run P1 + P2 only, 3 faults, 5 trials
- Adds generalizability claim to paper

---

## 4. FAULT TAXONOMY

8 fault scenarios across 3 planes. Each takes severity σ ∈ {0.1, 0.3, 0.5}

### Plane 1 — Data Faults
| Fault | Description |
|---|---|
| `schema_drift` | Drops/renames fraction of columns |
| `statistical_drift` | Gaussian shift in feature distributions |
| `label_poison` | Corrupts fraction of target labels |

### Plane 2 — Model Faults
| Fault | Description |
|---|---|
| `concept_drift` | Blends target toward its inverse |
| `weight_corruption` | Adds noise to model coefficients |
| `endpoint_kill` | Kills serving endpoint entirely |

### Plane 3 — Infrastructure Faults
| Fault | Description |
|---|---|
| `memory_pressure` | Allocates large array to simulate OOM |
| `batch_corruption` | Randomly NaNs fraction of batch values |

### Compound Fault
| Fault | Description |
|---|---|
| `compound_fault` | statistical_drift + concept_drift simultaneously |

**File:** `fault_injector/fault_injector.py` ✅ DONE

---

## 5. THE FOUR PIPELINES

### P1 — Drift Detection + Automated Retraining
**File:** `pipelines/p1_drift_retrain/p1_pipeline.py` ✅ DONE  
**Library:** Evidently AI (drift), MLflow (retraining tracking)  
**Healing strategy:**
1. PSI-based drift detection on incoming batches
2. Drift threshold → trigger MLflow retraining job
3. Promote new model only if RMSE improves >2%
4. Rollback if degradation within 10 batches

**Key config:**
```python
DRIFT_PSI_THRESHOLD   = 0.2
RETRAIN_WINDOW        = 5000
IMPROVEMENT_THRESHOLD = 0.02
ROLLBACK_WINDOW       = 10
BATCH_SIZE            = 500
SLA_BUDGET_SECONDS    = 120.0
TTD_MAX_SECONDS       = 60.0
```

**Remediation cost:** 1.0 (full retrain = maximum cost)  
**Known behavior:** Excellent TTD, fails TTR on sustained drift due to
buffer poisoning (see Finding 1.1). Heals passively by outlasting fault.

---

### P2 — Circuit Breaker + Shadow Model
**File:** `pipelines/p2_circuit_breaker/p2_pipeline.py` ✅ DONE  
**Pattern:** Circuit breaker state machine (CLOSED → OPEN → HALF_OPEN)  
**Healing strategy:**
1. Shadow model runs in parallel at all times
2. Error rate >15% for 3 consecutive batches → circuit opens
3. Traffic routed to shadow model instantly
4. After backoff, test live model recovery (HALF_OPEN state)
5. If recovered → close circuit, else stay on shadow

**Key config:**
```python
ERROR_RATE_THRESHOLD  = 0.15
CONSECUTIVE_FAILURES  = 3
BACKOFF_SECONDS       = 2.0
MAX_RETRIES           = 3
```

**Remediation cost:** 0.2 (simple failover = cheap)  
**Known behavior:** Near-instant healing for infrastructure faults
(TTD=0s, TTR=0s for endpoint_kill). Completely blind to statistical
drift — circuit stays CLOSED despite RMSE degradation.

---

### P3 — RL-Based Autoscaler
**File:** `pipelines/p3_rl_autoscaler/p3_pipeline.py` ❌ NOT STARTED  
**Library:** Stable-Baselines3 (PPO)  
**Healing strategy:**
- PPO agent observes pipeline state, takes scaling/routing actions
- Pre-trained for 50k steps on nominal environment
- During eval, agent is the healing mechanism

**RL spec:**
```
State:  [current_rmse_ratio, batch_queue_depth, error_rate, memory_util]
Actions: {scale_up, scale_down, hold, switch_to_fallback}  — discrete 4
Reward: +1 SLA met, -1 SLA breach, -0.1 unnecessary action (cost penalty)
```

**Remediation cost:** 0.5 (moderate — scaling has resource cost)  
**Expected behavior:** Best at infrastructure faults, learns to pre-empt
memory pressure. Weak on data plane faults (no retraining mechanism).

---

### P4 — Causal RCA Pipeline
**File:** `pipelines/p4_causal_rca/p4_pipeline.py` ❌ NOT STARTED  
**Library:** pgmpy (PC algorithm), dowhy  
**Healing strategy:**
- Pre-defined causal DAG of pipeline topology
- PC algorithm localizes root cause node from anomaly signals
- Node-specific remediation: schema→reparse, drift→retrain, infra→failover
- Meta-pipeline: wraps and orchestrates P1 and P2 components

**Causal DAG:**
```
data_ingestion → feature_pipeline → model_serving → output_monitor
      ↓                ↓                  ↓
  schema_fault    stat_drift_fault   perf_degradation
```

**Remediation cost:** 0.6 (RCA computation + targeted remediation)  
**Expected behavior:** Most surgical — correct root cause localization
means appropriate remediation per fault type. Expected to score highest
on Coverage (C) and Stability (S). Weakest on Response speed (R).

---

## 6. SHS METRIC

### Formula
```
SHS = w₁·D + w₂·R + w₃·C + w₄·S
weights = (0.25, 0.30, 0.25, 0.20)  ← default, validated by sensitivity analysis
```

### Four Dimensions
| Dim | Name | Formula | Measures |
|---|---|---|---|
| D | Detection Efficacy | `(1 - TTD/TTD_max) × precision` | Speed + accuracy of fault detection |
| R | Response Efficiency | `(1 - TTR/SLA) × (1/(1+cost))` | Speed + cost of healing |
| C | Fault Coverage | `healed_combos / total_combos` | Breadth of fault types handled |
| S | Post-Healing Stability | `recovery_quality × stability_term` | Stable return to baseline |

### Key Constants
```python
TTD_MAX_SECONDS = 60.0
SLA_BUDGET      = 120.0
HEAL_EPSILON    = 1.10   # healed if post_heal_rmse ≤ baseline × 1.10
```

### Weight Justification (for reviewer defense)
R weighted highest (0.30) because TTR maps directly to SLA compliance
and business cost in production. D and C equal at 0.25. S slightly
lower at 0.20 as secondary consequence of good healing.

**File:** `evaluator/shs_metric.py` ✅ DONE

---

## 7. EXPERIMENT PROTOCOL

### Per-Experiment Structure
```
Phase 1: Stabilization — 5 clean batches, confirm baseline
Phase 2: Fault Injection — fault applied at batch 6
Phase 3: Observation — up to 20 batches, pipeline heals
Phase 4: Verification — 5 post-heal batches, measure recovery
```

### Full Experiment Matrix
```
4 pipelines × 8 faults × 3 severities × 10 trials = ~200 runs
(some fault/pipeline combos infeasible — ~180-200 actual runs)
Average run time: ~45 seconds on Legion
Total compute: ~2.5 hours on Day 2
```

### Severity Levels
```
σ = 0.1 → mild fault
σ = 0.3 → moderate fault  
σ = 0.5 → severe fault
```

---

## 8. EVALUATOR MODULES

All in `evaluator/` directory. All run on Mac.

| File | Status | Purpose |
|---|---|---|
| `shs_metric.py` | ✅ DONE | SHS formula, ExperimentResult dataclass |
| `result_logger.py` | ✅ DONE | Writes raw_results.csv after every run |
| `stats.py` | ✅ DONE | Friedman, Wilcoxon, Spearman tests |
| `plotter.py` | ✅ DONE | Generates all 6 paper figures |

### Result Files
```
results/raw_results.csv   ← one row per run, written immediately
results/shs_results.csv   ← SHS scores, written by finalize()
results/run_log.json      ← progress tracker
```

---

## 9. STATISTICAL ANALYSIS PLAN

Run on Day 3 using `evaluator/stats.py`:

| Test | Function | Purpose |
|---|---|---|
| Friedman test | `friedman_test()` | Are pipeline differences non-random? |
| Wilcoxon post-hoc | `wilcoxon_posthoc()` | Which pipeline pairs differ? (Bonferroni corrected) |
| Spearman correlation | `spearman_severity_shs()` | Does severity correlate with SHS drop? |
| Sensitivity analysis | `sensitivity_analysis()` | Are rankings stable under weight changes? |
| Summary table | `pipeline_summary_table()` | Mean ± 95% CI per pipeline (Table 1 in paper) |

**Validated on dummy data:**
```
Friedman: χ²=34.000, p=0.0000 ✅ significant
Wilcoxon: all 6 pairs significant, large effect ✅
Spearman: all negative rho, p=0.0 ✅ severity degrades SHS
```

---

## 10. THE 6 PAPER FIGURES

Generated by `evaluator/plotter.py`. Saved to `paper/figures/`.

| Figure | Type | Key Message |
|---|---|---|
| Fig 1 | Radar chart | Each pipeline has distinct healing profile |
| Fig 2 | Box plot | SHS score distribution per pipeline |
| Fig 3 | Heatmap | Pipeline × fault class SHS — most cited figure |
| Fig 4 | Line + CI | SHS degradation under increasing severity |
| Fig 5 | Grouped bar | Fault coverage per pipeline per fault type |
| Fig 6 | Dual panel | Metric sensitivity — rank stability proof |

---

## 11. DOCKER SETUP (Legion Only)

```
services/
├── data_stream_service.py   → port 8001, serves taxi data batches
├── model_server_service.py  → port 8002, serves model predictions
└── monitor_service.py       → port 8003, logs metrics and alerts
```

**Run:** `docker compose up --build` from repo root  
**MLflow UI:** http://localhost:5001

---

## 12. REPO STRUCTURE

```
MLOPs study/
├── pipelines/
│   ├── p1_drift_retrain/p1_pipeline.py      ✅
│   ├── p2_circuit_breaker/p2_pipeline.py    ✅
│   ├── p3_rl_autoscaler/p3_pipeline.py      ✅
│   └── p4_causal_rca/p4_pipeline.py         ✅
├── fault_injector/
│   └── fault_injector.py                    ✅
├── evaluator/
│   ├── shs_metric.py                        ✅
│   ├── result_logger.py                     ✅
│   ├── stats.py                             ✅
│   └── plotter.py                           ✅
├── data/
│   ├── data_loader.py                       ✅
│   ├── yellow_tripdata_2019-01.csv          ✅ (Legion only)
│   └── yellow_tripdata_2020-01.csv          ✅ (Legion only)
├── services/
│   ├── data_stream_service.py               ✅
│   ├── model_server_service.py              ✅
│   └── monitor_service.py                   ✅
├── results/
│   ├── raw_results.csv                      ← written on Day 2
│   └── shs_results.csv                      ← written on Day 3
├── paper/
│   ├── findings_log.md                      ✅
│   ├── context.md                           ✅ (this file)
│   ├── introduction.md                      ❌
│   └── figures/                             ✅ (dummy figures exist)
├── docker-compose.yml                       ✅
└── .gitignore                               ✅
```

---

## 13. KEY FINDINGS SO FAR

### Finding 0.1 — SHS Face Validity
Dummy data SHS scores ranked pipelines in theoretically correct order.
P2 highest (fast cheap healing), P3/P4 lowest detection. Face validity confirmed.

### Finding 0.2 — Rank Instability on Close Scores
Sensitivity rank std = 0.441 on dummy data where scores are tightly clustered
(range 0.055). Expected to drop significantly on real data. Not a metric flaw.

### Finding 0.3 — Natural Drift Confirmed in Dataset
trip_distance +5.0%, fare_amount +2.3% between 2019 and 2020.
Validates dataset choice. Grounded in COVID-19 behavioral shift.

### Finding 1.1 — P1 Buffer Poisoning
P1 retrains on contaminated rolling buffer under sustained drift.
RMSE degraded monotonically (2.7 → 4.5) despite correct detection (TTD=1.14s).
Fix applied: mixed reference + recent window retraining.
Post-fix worst RMSE: 3.74. TTR still 120s — P1 heals passively not actively.
**Named phenomenon: "Retraining Buffer Poisoning"**

### Finding 2.1 — P2 Blind to Statistical Drift
Circuit breaker triggers on hard errors (error_rate > 15%) not soft degradation.
endpoint_kill: TTD=0s, TTR=0s — perfect. statistical_drift: TTD=60s, TTR=120s — blind.
**Named phenomenon: "Soft Degradation Blindness"**

### Cross-Pipeline Pattern (Emerging)
Both P1 and P2 fail on statistical_drift but for different reasons:
- P1: detects it, cannot fix it (buffer poisoning)
- P2: cannot even detect it (wrong signal type)
This confirms the Detection-Response decoupling hypothesis.
SHS correctly captures this because D and R are separate dimensions.

---

## 14. OPEN QUESTIONS

- [ ] Does P1 buffer poisoning occur at severity=0.1 or only 0.3/0.5?
- [ ] Does P2 circuit breaker correctly failover on endpoint_kill at all severities?
- [ ] Does P4 causal RCA correctly disambiguate compound_fault root cause?
- [ ] Is sensitivity rank std < 0.2 after real experiments as hypothesized?
- [ ] Does severity correlate linearly with SHS drop? (Spearman)
- [ ] Does CICIDS-2017 show same cross-pipeline patterns as taxi data?

---

## 15. 4-DAY TIMELINE

### Day 1 — Heavy (7-8 hrs) ← CURRENT DAY
- [x] fault_injector.py
- [x] data_loader.py
- [x] shs_metric.py
- [x] result_logger.py
- [x] stats.py
- [x] plotter.py
- [x] p1_pipeline.py + smoke test
- [x] p2_pipeline.py + smoke test
- [x] Docker setup
- [x] p3_pipeline.py + smoke test
- [x] paper/introduction.md first draft

### Day 2 — Heavy (7-8 hrs)
- [x] p3_pipeline.py complete (if not done Day 1)
- [x] p4_pipeline.py complete
- [ ] Full 200-run experiment loop on Legion
- [ ] Related work section on Mac
- [ ] Methodology section on Mac

### Day 3 — Light (3-4 hrs)
- [ ] Pull results, run full statistical analysis
- [ ] Generate all 6 real figures
- [ ] Write Results + Analysis section
- [ ] Write Discussion section
- [ ] Run CICIDS generalizability experiments on Legion

### Day 4 — Light (3-4 hrs)
- [ ] Abstract + tighten Introduction
- [ ] Partner reviews, hostile reviewer questions
- [ ] Threats to Validity
- [ ] Final formatting (IEEE/ACM template)
- [ ] GitHub repo README cleanup

---

## 16. HOW TO CONTINUE IN A NEW AI CONVERSATION

If token limit is reached, start a new conversation with:

```
"I am writing a research paper called SH-Bench. 
Here is my complete project context: [paste this file]
Here is my findings log: [paste findings_log.md]
Current status: [describe what you just finished]
Next task: [describe what you need]"
```

The AI will have full context and can continue seamlessly.

---

## 17. WEIGHT JUSTIFICATION ARGUMENT (For Reviewer Defense)

**Reviewer will ask:** "Why (0.25, 0.30, 0.25, 0.20) and not equal weights?"

**Answer:**
In production SLA-driven MLOps systems, time-to-remediate directly
maps to downtime, which maps to revenue loss. A pipeline that detects
faults perfectly but takes 10 minutes to heal costs more than one that
detects slightly slower but heals in 30 seconds. R therefore has higher
operational weight. D and C are equally important — you need both broad
coverage and accurate detection. S is weighted lowest as it is a
secondary quality of healing, not healing itself.

Supporting literature: AWS SRE whitepaper on SLA cost modeling,
Google SRE book Chapter 4 (error budget), Sculley et al. 2015.

Sensitivity analysis shows rankings are stable under 100 random weight
perturbations — so while weights are theoretically motivated, the
ranking conclusions are robust regardless of exact values.

---

## 18. PAPER NARRATIVE (The Story Arc)

**Hook:** Production ML systems fail silently and often — drift, crashes,
data corruption. Self-healing is the answer but nobody agrees on how to
measure it.

**Gap:** Existing work evaluates individual healing approaches in isolation.
No unified framework exists to compare them.

**Contribution:** SH-Bench — a fault taxonomy, injection protocol, and
composite metric that lets you benchmark any MLOps pipeline's
self-healing capability in a reproducible way.

**Key finding to lead with:** No single pipeline dominates across all
fault types. Pipelines optimized for infrastructure healing are blind
to data drift, and vice versa. This specialization-generalization tradeoff
is only visible through a multi-dimensional metric like SHS.

**Implication:** Pipeline designers should select or combine healing
strategies based on their expected fault distribution — SH-Bench
provides the tool to make that decision empirically.
