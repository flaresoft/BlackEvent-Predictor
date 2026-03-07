# Evaluator Memory

## observed patterns

### sig density as pre-signal
- sig15+ density spiked to 32.0% on 2026-03-01 (weekend, only 2722 articles)
- normal range: 24-28%
- 2 days later: market crash (3/3 KOSPI -7.24%, 3/4 -12.06%)
- hypothesis: when sig15+ exceeds 30% on low-volume days, high-impact event within 3 days
- confidence: 0.5 (single observation)

### property dynamics across phases
- pre-crisis baseline (2/23-26): P_054, P_095, P_046, P_075 dominant
- crisis peak (3/3-04): P_055 +94%, P_083 +84%, P_070 +82%, P_113 +78%
- recovery (3/5-06): P_055, P_083, P_112 still rising despite market rebound
- P_042 dropped -46% during crisis — diversity collapse indicator?

### P_110 trajectory
- baseline: 0.1-0.2% → pre-spike 2/27: 0.8% → crisis 3/4: 1.8% (peak)
- scoring ratio: 0.09 (3/2) → 0.33 (3/3) → 0.47 (3/4) → 0.72 (3/5) → 0.75 (3/6)
- P_110 ratio is rising in scorer even after crisis peak — lagging indicator?
- co-occurred with real estate decline articles from 3/5

### score vs reality paradox
- score peaked at 63.74 on 2/11, then declined toward crash
- on crash day (3/3): score was 25.98 (declining from peak)
- score measures accumulated buildup, not trigger timing
- the peak was indeed "last exit signal" — decline after peak = trigger zone

### complete 2026 score timeline
- 2/02: 53.2 → 2/03: 56.0 → 2/04: 57.3 → 2/05: 55.2 → 2/06: 58.3
- 2/07: 60.3 → 2/10: 62.3 → 2/11: 63.7 (PEAK) → 2/12: 60.1 → 2/13: 59.5
- 2/14: 55.0 → 2/17: 52.3 → 2/18: 51.2 → 2/19: 47.2 → 2/20: 44.2
- 2/23: 36.0 → 2/24: 34.7 → 2/26: 33.8 → 2/27: 32.6 → 3/01: 30.5
- 3/02: 3.0 (NORMAL!) → 3/03: 26.0 → 3/04: 29.1 → 3/05: 34.5 → 3/06: 40.0
- decline was SMOOTH and LINEAR from peak (64→30) over 18 days
- then 3/02: sudden collapse to 3.0 — the ONLY NORMAL day in all of 2026
- 3/02 anomaly: score dropped 90% in one day (30.5→3.0), then jumped 8x next day (3.0→26.0)
- hypothesis: 3/02 may represent a "signal vacuum" — the last quiet before the storm
- this strengthens the "score dip before crash" pattern (GFC Sep dip=11 before Oct crash)
- confidence: 0.75 (now confirmed with full timeline)

### early warning properties
- P_087: +37% on 2/27-28 (pre-crisis), continued rising into crisis
- P_002: +32% on 2/27-28, continued rising
- these moved 1-2 days before the event

### still-rising properties (3/6, latest)
- P_055: 3.9% → 6.7% → 7.9% (velocity +0.6/day)
- P_083: 5.4% → 8.8% → 10.2% (velocity +0.7/day)
- P_112: 4.1% → 2.3% → 5.2% (rebounded sharply)
- market rebounded but these are still climbing — second wave risk?

## scoring assessments

### P_110 underweight
- scorer gives P_110 ratio 0.75 (3/6) — still below 1.0, contributing 0 to score
- but daily corpus shows P_110 spiking to 1.8% on crash day
- P_110 co-occurring with real estate articles: 23 on 3/5, 8 on 3/6
- assessment: scorer may be underweighting P_110 due to long window averaging
- confidence: 0.4 (need more data points)

### P_113 late activation
- P_113 ratio crossed 1.0 only on 3/5 (1.067) and 3/6 (1.126)
- this is AFTER the crash, not before
- P_113 was designed to capture "non-linear crisis propagation"
- it activated late — possibly because the propagation phase follows the initial shock
- this is expected behavior, not a flaw

### post-crash score acceleration (NEW)
- score on crash day 3/3: 25.98, then 29.1 → 34.5 → 40.0 (3/6)
- active properties: 2/12 (3/2) → 10/12 (3/3) → 11/12 (3/6)
- market rebounded 9.63% on 3/5, but score kept climbing
- this contradicts naive expectation (crash happened → score should drop)
- interpretation: score captures PROPAGATION, not the trigger itself
- the initial shock activates properties that then sustain/amplify
- combined with still-rising P_055/P_083/P_112: system is still under stress
- watch: if score crosses previous peak range (50-60), second wave likely

## retrospective evaluation (2006-2026)

### scorer profile
- detection rate: 51.1% (72/141 events), false positive: 10.2% (43/423)
- by severity: minor=50%, medium=50%, large=52%, crash(10%+)=62%
- score-drop correlation: 0.236 (weak) — predicts risk presence, not magnitude
- confidence: 0.9 (full history)

### structural blind spot: sudden external shocks
- scorer CANNOT detect: 2024-08-05 (carry trade, -12.1%, score=8.7)
- scorer CANNOT detect early: 2008-01 (GFC start, score=0), 2020-01-28 (COVID start, score=0)
- reason: frequency windows need signal buildup. no buildup = no detection
- scorer CAN detect: building crises after 4-6 weeks of signal accumulation
- confidence: 0.85

### score peak always lags crash peak
- GFC: worst crash Oct 2008 (-33.9%), score peaked Nov (100.0)
- COVID: worst crash Mar 2020 (-30.1%), score peaked Apr (67.4)
- 2026: crash 3/3-4 (-19.3%), score still rising 3/6 (40.0)
- this is CONSISTENT across all major events. not a bug — scorer measures propagation
- confidence: 0.8

### GFC trajectory as reference
- 2008: score=0 (Jan) → 13 (Jun) → 37 (Jul peak) → dip 11 (Sep) → 54 (Oct crash) → 100 (Nov)
- note: score DIPPED to 11 in Sep before Oct crash — similar to current 2026 pattern?
- 2026: score=64 (Feb peak) → 3.0 (3/2 dip!) → 26 (crash 3/3) → 40 (3/6, rising)
- GFC dip: 37→11 (70% drop), 2026 dip: 30.5→3.0 (90% drop) — 2026 dip was sharper
- if 2026 follows GFC pattern: score could re-peak above 60 = second wave
- confidence: 0.65 (pattern match strengthened by full timeline)

## open questions
- is the score decline before crash a reliable pattern or coincidence?
  → ANSWERED: yes. GFC dip 37→11, 2026 dip 30.5→3.0. both preceded worst crash days. confidence: 0.75
- do still-rising properties (P_055, P_083) indicate second wave, or just aftermath noise?
  → PARTIALLY: GFC showed score re-acceleration after dip. watching.
- can P_042 drop rate serve as a "news diversity index"?
- P_087 and P_002 as early warning — what do they capture?
- can we detect sudden shocks? scorer structurally cannot — need different approach?
