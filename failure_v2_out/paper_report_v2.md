# Failure Analysis v2: Direct vs CoT RAG

## Taxonomy (Primary labels)

- **R-GAP**: Retrieval Gap (gold evidence absent)
- **R-NOISE**: Retrieval Noise (mostly irrelevant)
- **R-TRAP**: Retrieval Trap (supports wrong option)
- **U-FAIL**: Evidence Underuse (gold present, not used)
- **I-FAIL**: Inference Failure (gold decisive, still wrong)
- **AMBIG**: Ambiguity / underspecified
- **ERROR**: LLM/API error

## Overview

- **SET1 (Direct wrong, CoT correct)**: 87
- **SET2 (Both wrong)**: 200

## SET1 Distribution

| Category | Count | Percentage |
|---|---:|---:|
| R-GAP | 0 | 0.00% |
| R-NOISE | 48 | 55.17% |
| R-TRAP | 4 | 4.60% |
| U-FAIL | 4 | 4.60% |
| I-FAIL | 29 | 33.33% |
| AMBIG | 2 | 2.30% |
| ERROR | 0 | 0.00% |

## SET2 Distribution

| Category | Count | Percentage |
|---|---:|---:|
| R-GAP | 0 | 0.00% |
| R-NOISE | 176 | 88.00% |
| R-TRAP | 24 | 12.00% |
| U-FAIL | 0 | 0.00% |
| I-FAIL | 0 | 0.00% |
| AMBIG | 0 | 0.00% |
| ERROR | 0 | 0.00% |

## Representative Examples

### R-NOISE — Retrieval Noise (mostly irrelevant)

**SET1 | QID 867**
- Q: A 21-year-old man comes to the emergency room with swelling and severe pain in his left lower leg that started 2 hours ago. He has no history of serious illness or trauma. His father has a history of pulmonary embolism. ...
- primary=R-NOISE  secondary=None  rescue=PARAMETRIC_KNOWLEDGE_RESCUE

**SET2 | QID 1089**
- Q: A 42-year-old man presents to your office complaining of right-sided facial swelling that has progressively worsened over the last month after returning from a trip to India. On examination, the patient has an obvious di...
- primary=R-NOISE  secondary=D=R-NOISE|C=R-NOISE

### R-TRAP — Retrieval Trap (supports wrong option)

**SET1 | QID 998**
- Q: A 72-year-old woman with a 40 pack-year history of smoking presents to your office with jaundice. After a thorough workup, you determine that the patient has pancreatic cancer. Which of the following is the most appropri...
- primary=R-TRAP  secondary=None  rescue=PARAMETRIC_KNOWLEDGE_RESCUE

**SET2 | QID 1250**
- Q: A 37-year-old woman with an HIV infection comes to the physician for a follow-up examination. Six months ago, combined antiretroviral therapy consisting of dolutegravir, tenofovir, and emtricitabine was initiated. Labora...
- primary=R-TRAP  secondary=D=R-TRAP|C=R-NOISE

### U-FAIL — Evidence Underuse (gold present, not used)

**SET1 | QID 475**
- Q: A 63-year-old man comes to the emergency department because of retrosternal chest pain. He describes it as 7 out of 10 in intensity. He has coronary artery disease, hypertension, and type 2 diabetes mellitus. His current...
- primary=U-FAIL  secondary=None  rescue=PARAMETRIC_KNOWLEDGE_RESCUE
- Direct gold quote: [First_Aid_Step2_44] ST deviation ≥ 0.5 mm 1 + cardiac marker

### I-FAIL — Inference Failure (gold decisive, still wrong)

**SET1 | QID 1080**
- Q: A 5-day-old boy is brought to the emergency department because of a 1-day history of poor feeding, irritability, and noisy breathing. The mother did not receive any prenatal care. His respirations are 26/min. Physical ex...
- primary=I-FAIL  secondary=None  rescue=PARAMETRIC_KNOWLEDGE_RESCUE
- Direct gold quote: [InternalMed_Harrison_12016] tetanus caused more than 1 million deaths

### AMBIG — Ambiguity / underspecified

**SET1 | QID 596**
- Q: Six hours after birth, a newborn boy is evaluated for tachypnea. He was delivered at 41 weeks' gestation via Caesarian section and the amniotic fluid was meconium-stained. His respiratory rate is 75/min. Physical examina...
- primary=AMBIG  secondary=None  rescue=PARAMETRIC_KNOWLEDGE_RESCUE
- Direct gold quote: [First_Aid_Step2_964] cyanosis, nasal flaring, intercostal retractions
