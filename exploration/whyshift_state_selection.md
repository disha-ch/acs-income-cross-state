# WhyShift evidence for ACS state selection

## Paper evidence

The paper studies ACSIncome geographic transfer with California as the source and Puerto Rico as a target. In its selected CA → PR example, DISDE attributes **85.4% of the measured performance degradation to a change in P(Y | X)** (Figure 3(a)). The paper's case study reports a raw XGBoost accuracy drop of about **10.3 percentage points** (Figure 10(a)). This is direct conditional-shift evidence for that exact pair and task, not for every low-prevalence state. The paper uses the income-over-$50K target and reports XGBoost for the DISDE decomposition.

| Question | Evidence-based answer |
|---|---|
| Which state pair? | California → Puerto Rico (ACSIncome, geographic transfer). |
| Was transfer difficult? | Yes for the paper's reported XGBoost case: Figure 10(a) labels a raw accuracy drop of about 10.3 percentage points. |
| Measured source-to-target accuracy gap | About 10.3 percentage points in the paper's CA → PR case study. Treat this as the figure's rounded result; it is not a result from this project. |
| Was the difficulty conditional? | Yes. Figure 3(a) reports a DISDE Y|X-shift ratio of 85.4%. DISDE estimates the conditional component while controlling for feature composition through a shared covariate distribution. |
| Evidence controlling for feature composition | The paper's DISDE method compares conditional risks over shared X-support and attributes the remaining component to Y|X shift; Appendix B.1 defines the ratio. |
| Is it relevant to our hidden-state selection? | It establishes that ACSIncome can have substantial conditional shift across geographic domains. Puerto Rico is outside our six-state hidden set, so it does not validate MS, WV, NM, AR, LA, or MT individually. |

**Scope and design mismatch:** the cited CA → PR experiment trains on California alone. This project trains on a pooled CA, TX, and NY configuration. Its measured gap and DISDE attribution therefore cannot be transferred directly to the pooled-source model or used to infer target difficulty for our six hidden states. Puerto Rico is a U.S. territory, not one of the 50 states covered by this project's current exploration, so it should not be substituted into the state-based hidden set automatically.

WhyShift also evaluates CA as a source across 50 target domains and reports that, among the 14 CA-source pairs with accuracy degradation above 5 percentage points, 13 had more than half of degradation attributed to Y|X shift. The published summary does not identify those 14 target states in the evidence reviewed here, so this supports the general risk of conditional shift, not a ranking of our six targets.

## Provisional-state evidence from local EDA

The state/year summary gives these unweighted positive-label rates and retained row counts:

| State | 2018 rows | 2018 positive rate | 2021 rows | 2021 positive rate |
|---|---:|---:|---:|---:|
| MS | 13,189 | 0.2695 | 12,500 | 0.2964 |
| WV | 8,103 | 0.2906 | 7,488 | 0.3427 |
| NM | 8,711 | 0.2977 | 8,591 | 0.3647 |
| AR | 13,929 | 0.2680 | 13,554 | 0.3109 |
| LA | 20,667 | 0.3325 | 19,821 | 0.3651 |
| MT | 5,463 | 0.2925 | 5,408 | 0.3572 |

These figures show usable retained samples and marginal label-rate differences. They do **not** measure transfer accuracy, conditional shift, or rank which target is hardest. In particular, MS is the paper's source state for an ACS Mobility → Hawaii example; that is a different prediction task and does not establish MS as a difficult ACSIncome target.

## Decision and next evidence needed

Keep MS, WV, NM, AR, LA, and MT labeled **provisional geographic evaluation targets**. The cited evidence demonstrates a large, predominantly conditional-shift performance drop for CA → PR; it does not demonstrate that these six states are high-conditional-shift targets. It also does not match this project's pooled CA/TX/NY training configuration. To validate the six states for ACSIncome, evaluate each intended source → target pair with the project feature contract and fixed model protocol; report source and target accuracy, the raw gap, DISDE's Y|X share, uncertainty intervals, and sample counts. Apply the same protocol and preprocessing to every pair. A large positive-label-rate difference alone is insufficient.

## Second held-out instance

The repository already has a candidate temporal replicate: `tests/hidden_data/hidden_2018.npz` and `hidden_2021.npz`, each containing all six provisional states. The local archives currently contain 70,062 (2018) and 67,362 (2021) rows, with 10 features and a separate state ID per row; the smallest state-year cell has 5,408 rows. This provides a second year-specific evaluation instance under the existing feature contract and is not a renamed copy of the 2018 archive.

The project-root diagnostic passed:

```text
2018 rows: 70062 states: ['AR', 'LA', 'MS', 'MT', 'NM', 'WV']
2021 rows: 67362 states: ['AR', 'LA', 'MS', 'MT', 'NM', 'WV']
PASS: Both evaluation years cover the same six states
```

The instances differ by ACS survey year and sampled rows while sharing state coverage and the 10-feature contract. They can be scored independently in principle by applying the same frozen model and metric code to each archive, then reporting 2018 and 2021 results separately. The current repository does not yet demonstrate that behavior: `instruction.md` is still the Otter template, and `tests/checks.py` contains unimplemented placeholder checks. Thus the archives pass the coverage diagnostic, but the task's dual-split scoring requirement is not yet implemented or verified. The diagnostic says nothing about person-level independence.

Before calling the instances record-disjoint, verify source-row uniqueness across the two annual PUMS extracts using a protected local audit key and confirm no duplicated source rows; do not publish or package that key. Annual ACS samples do not provide a persistent public person identifier, so this check establishes non-duplication of source records, not that a person could never be sampled in multiple years. If the benchmark requires strict person-level disjointness, the available public identifiers cannot prove that guarantee. Do not create another archive until the verifier's instance semantics and this independence check are clear.

## Sources

- Wang et al., *Rethinking Distribution Shifts: Empirical Analysis and Modeling for Tabular Data*, arXiv:2307.05284, especially Table 1, Figure 3(a), Figure 10(a), and Appendix B.1: [paper PDF](https://arxiv.org/pdf/2307.05284).
- WhyShift implementation and benchmark setup: [official repository](https://github.com/namkoong-lab/whyshift).
- Local state/year counts and rates: `exploration/eda_outputs/state_year_summary.csv`.
