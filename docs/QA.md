# MindScan — Professor Cross-Examination Prep

> Covers all rubric areas: Abstract, Related Work, Datasets, Methodology, Evaluation, Conclusions, Architecture, and Gotcha questions.
> Numbers are verified against notebook code and the dashboard.

---

## Quick-Reference Numbers Table

| Metric | Value | Source |
|---|---|---|
| Total models | 12 (4×3 datasets) | Architecture |
| Datasets | 3 (D1, D2, D3) | Data section |
| D1 — train/test split | 80/20 | Notebook |
| D1 — class imbalance | 1.89× | Notebook |
| D2 — class imbalance | 3.46× | Notebook |
| D3 — class imbalance | Pre-balanced | Notebook |
| D1 best model (classical) | Logistic Regression 93.4% | Dashboard |
| D1 XLM-RoBERTa | 92.3% | Dashboard |
| D2 best model (classical) | SVM 97.1% | Dashboard |
| D2 XLM-RoBERTa | 96.9% | Dashboard |
| D3 best model (classical) | Random Forest 79.5% | Dashboard |
| D3 XLM-RoBERTa | 82.4% | Dashboard |
| D3 XGBoost (collapsed) | 71.0% | Dashboard |
| TF-IDF max_features | 50,000 | predict.py / notebooks |
| TF-IDF ngram_range | (1,2) | notebooks |
| TF-IDF sublinear_tf | True | notebooks |
| TF-IDF min_df | 2 | notebooks |
| XLM-RoBERTa parameters | 278M | HuggingFace model card |
| XLM-RoBERTa lr | 2e-5 | notebooks |
| XLM-RoBERTa epochs | 3 | notebooks |
| XLM-RoBERTa max_length D1/D2 | 128 tokens | notebooks |
| XLM-RoBERTa max_length D3 | 256 tokens | notebooks |
| D3 SMOTE applied? | No (pre-balanced) | Notebook |
| risk_flag majority threshold | ≥3/4 D3 models | JS logic |
| Inference backend | HuggingFace Space | app.py |
| Proxy URL | esvanth-mindscan.hf.space | app.py |
| Text length limit | 5,000 characters | app.py |

---

## Section 1 — Abstract & Objectives

**Q1. What is the core research question?**
Can an ensemble of 12 NLP models — spanning 3 distinct suicide/depression datasets and two architectures (classical ML + transformer) — produce a more robust mental health risk signal than any single model alone?

**Q2. What gap in the literature does this address?**
Most prior work trains on a single dataset, limiting generalization. We show that cross-dataset ensemble coverage exposes domain-specific biases (e.g., Twitter Affect vs. Clinical Lexicon) that are invisible when evaluating on a held-out split from the same source.

**Q3. What are the three datasets and what do they represent?**
- D1: Depressive vs. non-depressive text (Reddit/general social media)
- D2: Suicide vs. non-suicide Twitter posts
- D3: Multi-class (suicide, depression, anxiety, normal) Reddit posts

**Q4. Why run all 12 models in parallel rather than picking the best one?**
Disagreement between models is itself a signal. When classical TF-IDF models agree but XLM-RoBERTa disagrees, the ensemble flags "Ensemble Conflict" rather than a confident prediction — the safe, clinical-conservative choice.

**Q5. What is the final risk decision rule?**
`risk_flag` = majority vote among 4 D3 models (≥3/4 say "suicide"). The XLM-RoBERTa D3 output is the tiebreaker for the "Masked Suicidality" clinical insight.

**Q6. What do you mean by "parallel ensemble"?**
All 12 models receive the same raw text simultaneously. There is no sequential filtering or gating. Results from all three datasets are surfaced independently, then a meta-level consensus is computed.

**Q7. Does this system replace clinical diagnosis?**
No — explicitly stated in the dashboard. This is a research screening tool. Any positive flag should prompt human clinical review.

---

## Section 2 — Related Work

**Q8. What is the baseline you compare against?**
We compare against Tumaliuan et al. (2023), who report SVM accuracies on similar Reddit data. Our D3 SVM reaches 77.8%, consistent with their reported range. XLM-RoBERTa at 82.4% improves on that baseline.

**Q9. What is MentalBERT and why didn't you use it?**
Ji et al. (2022) released MentalBERT/MentalRoBERTa pretrained on mental health forum data. We used general-domain XLM-RoBERTa because: (a) MentalBERT is English-only and we prioritized multilingual capability; (b) compute/timeline constraints. Acknowledged as future work.

**Q10. What is the MTL precedent for your architecture?**
Zogan et al. (2024) showed multi-task learning across depression/suicide signals outperforms single-task models. Our parallel ensemble is a late-fusion approximation — we don't share intermediate representations but achieve similar cross-signal coverage.

**Q11. What is the Affective vs. Clinical Lexicon Gap?**
Documented in NAACL 2024: models trained on social media affect language fail to generalize to clinical presentations using anhedonia/fatigue/hopelessness vocabulary. Our D2 (Twitter) models show exactly this — they under-flag clinical-style input. This is Finding 04 in the dashboard.

---

## Section 3 — Dataset & Methods

**Q12. How did you handle class imbalance?**
D1 and D2 had imbalanced classes (1.89× and 3.46× respectively). We applied SMOTE to training sets only — never the test set. D3 was pre-balanced and received no SMOTE.

**Q13. Why SMOTE and not class weighting?**
SMOTE creates synthetic feature-space neighbors, more effective for TF-IDF high-dimensional spaces than scalar weighting. Class weighting was also tested; SMOTE showed equal or better macro-F1 in cross-validation.

**Q14. What was the train/test split strategy?**
Stratified 80/20 split for all three datasets, preserving class proportions. No cross-dataset evaluation — models are dataset-specific by design.

**Q15. Why not use cross-validation for final evaluation?**
Compute constraints on transformer models. Classical models were validated with 5-fold CV during hyperparameter tuning. Final reported numbers are on the held-out 20% test set.

**Q16. How did you vectorize text for classical models?**
TF-IDF with `max_features=50,000`, `ngram_range=(1,2)`, `sublinear_tf=True`, `min_df=2`. Unigrams + bigrams capture local phrasing; sublinear TF dampens frequency dominance; min_df=2 filters hapax legomena.

**Q17. What preprocessing did you apply?**
Lowercasing, URL removal, mention/hashtag normalization. No stemming/lemmatization — TF-IDF vocabulary is token-level and stemming degrades n-gram quality. XLM-RoBERTa uses its own SentencePiece subword tokenizer.

**Q18. Why is D3 max_length 256 vs 128 for D1/D2?**
D3 Reddit posts are longer on average vs. Twitter-length D2 posts. 256 tokens captures more context for the multi-class classification task.

---

## Section 4 — Methodology & CRISP-DM

**Q19. What CRISP-DM phases does your pipeline cover?**
All 6: Business Understanding (RQs), Data Understanding (EDA), Data Preparation (preprocessing + SMOTE), Modeling (12 models), Evaluation (accuracy, F1, Kappa), Deployment (Flask + HuggingFace Space).

**Q20. How does your 3-step pipeline map to CRISP-DM?**
- Step 1 "Ingest & Vectorize" = Data Understanding + Data Preparation
- Step 2 "Train 12 Models" = Modeling
- Step 3 "Ensemble Vote" = Evaluation + Deployment

**Q21. Why four model types per dataset?**
Logistic Regression (linear baseline), SVM (margin-based), Random Forest (non-linear ensemble), XGBoost (boosted trees) + XLM-RoBERTa (transformer). Each captures different inductive biases. If all agree, confidence is high; divergence flags uncertainty.

**Q22. How was XLM-RoBERTa fine-tuned?**
Standard sequence classification fine-tuning: Adam optimizer, lr=2e-5, 3 epochs, batch size 16, linear warmup scheduler. Cross-entropy loss. Best checkpoint by validation accuracy.

**Q23. What is the role of XLM-RoBERTa vs. classical models?**
Classical models are fast and interpretable (feature importances accessible). XLM-RoBERTa provides contextual understanding that lexical models miss — especially for negation, irony, and clinical vocabulary. Classical models provide breadth; XLM-RoBERTa provides depth.

**Q24. Is there any data leakage?**
No. SMOTE is applied after the train/test split, only to training data. TF-IDF vocabulary is fit on training data only. XLM-RoBERTa tokenizer is fixed (pretrained). No test data was seen during any training step.

**Q25. Did you tune hyperparameters?**
Classical models: grid search with 5-fold CV on training set (C for LR/SVM, n_estimators/max_depth for RF/XGB). XLM-RoBERTa: lr and epochs from literature defaults (2e-5, 3 epochs) with validation monitoring.

**Q26. What is the significance of XGBoost "collapsing" on D3?**
XGBoost achieved only 71.0% on D3 — lowest classical result. Boosted trees overfit to majority class patterns when TF-IDF feature space has vocabulary overlap between classes (suicidal vs. depressive language share significant lexical overlap). This is the TF-IDF Lexical Overfitting finding.

**Q27. What is "Masked Suicidality" as a clinical insight?**
When D3 XLM-RoBERTa flags suicide risk AND D1 models say "non-depressive" — the person may be masking depression behind neutral or positive language. Aligns with clinical literature on "smiling depression" and active suicidal ideation without overt depressive symptoms.

---

## Section 5 — Evaluation & Presentation

**Q28. Why does the dashboard show accuracy as the primary metric?**
Accessibility — the research audience includes non-specialists. The IEEE technical report includes Macro F1-Score, Cohen's Kappa, and per-class precision/recall. The dashboard notes: "Full performance evaluation including Macro F1-Score, Cohen's Kappa... in IEEE Report."

**Q29. What is Cohen's Kappa and why does it matter here?**
Kappa measures inter-rater agreement corrected for chance. A model always predicting the majority class can achieve high accuracy but low Kappa. Kappa > 0.8 indicates strong agreement beyond chance.

**Q30. What is Macro F1 and why use it over Micro F1?**
Macro F1 averages F1 per class without weighting by support, penalizing models that ignore minority classes. Given imbalanced suicide/depression data, Macro F1 is the appropriate primary metric for research reporting.

**Q31. How do you interpret the Evidence Matrix?**
Each row = dataset, each column = model. Red cells = performance collapse. Green cells = strong performance. The pattern shows XLM-RoBERTa is consistently competitive or best, while XGBoost is the most volatile classical model.

**Q32. What does the D3 column tell you architecturally?**
D3 is the hardest task (4-class). Classical model performance is lower and more variable. XLM-RoBERTa's relative advantage is largest here — evidence that contextual representations matter most for fine-grained classification.

**Q33. Is the live demo connected to real models?**
Yes — Flask proxies to HuggingFace Space (`esvanth-mindscan.hf.space`) running `predict.py` with 12 loaded models. No hardcoded responses exist in the demo flow.

**Q34. What happens when the HuggingFace Space is sleeping?**
A 504 timeout is returned after 120 seconds with user-facing message: "HuggingFace Space timed out — it may be waking up, try again in 30s." The Space auto-wakes within ~60 seconds.

**Q35. Can you explain the three banner states?**
- Red "High Suicide Risk": D3 majority vote + XLM-RoBERTa agree on suicide
- Amber "Ensemble Conflict": Classical D3 models flag risk but XLM-RoBERTa disagrees
- Green "Low Risk": No majority risk signal

**Q36. Why is the amber state clinically important?**
Prevents false alarms while surfacing uncertainty. A pure majority vote without XLM-R agreement could trigger red alerts on metaphorical language ("I'm dying of embarrassment"). Amber flags for review without over-alarming.

---

## Section 6 — Conclusions & Future Work

**Q37. What are your two main research conclusions?**
- RQ1: Parallel ensemble reduces single-model failure modes. No single model dominates across all three datasets; the ensemble catches cases individual models miss.
- RQ2: Cross-dataset coverage reveals domain generalization failures invisible in single-dataset evaluation — specifically Twitter Affect Bias (D2) and TF-IDF Lexical Overfitting (D3).

**Q38. What are the two main limitations?**
- Lim1: Affective vs. Clinical Lexicon Gap — D2 Twitter models fail on clinical-register text. Needs domain-adapted pretraining (MentalBERT direction).
- Lim2: TF-IDF Lexical Overfitting on D3 — vocabulary overlap between depressive and suicidal language causes false positives. XLM-RoBERTa mitigates but doesn't solve this.

**Q39. What is the single most important future direction?**
Replace TF-IDF classical models with MentalBERT/MentalRoBERTa fine-tuned on all three datasets in a true multi-task learning setup (shared encoder, task-specific heads). This addresses both limitations simultaneously.

---

## Section 7 — Technical & Architecture

**Q40. Where does the prediction logic live?**
`predict.py` (LOCAL mode) or proxied to HuggingFace Space (PROXY mode). Flask `app.py` auto-detects mode at startup by checking if `models/classical/` directory exists.

**Q41. What is the structure of the predict endpoint response?**
JSON with: per-dataset results (accuracy, prediction, confidence), `risk_flag` (bool), `suicide_votes` (int, D3 majority count), `xlmr_d3` (transformer vote), `processing_time_ms`.

**Q42. Why Flask and not FastAPI?**
Project timeline and team familiarity. Flask is sufficient for a research demo with low concurrency. FastAPI would be preferred for production (async, OpenAPI docs, Pydantic validation).

**Q43. What are the security considerations for the text input?**
Server-side: max 5,000 character limit enforced in `app.py`. Empty string rejected. No SQL — no injection surface. Content is not stored. HuggingFace proxy is over HTTPS.

**Q44. Could this be adversarially manipulated?**
Yes — a motivated user could craft text avoiding trigger vocabulary while conveying suicidal intent. This is a known limitation of lexical models, partially why the transformer is included. Robustness testing is future work.

**Q45. How would you scale this to production?**
ONNX-exported models behind a FastAPI server, Redis caching for repeated inputs, horizontal scaling behind a load balancer. The parallel architecture is naturally parallelizable with asyncio or threading.

---

## Section 8 — Gotcha / Trick Questions

**Q46. Your D2 SVM gets 97.1% — isn't that suspiciously high?**
Twitter suicide datasets tend to have high separability because suicidal posts often contain explicit keywords. This is a known artifact of Twitter-scraped data. Accuracy drops on clinical-register text — which is exactly the Twitter Affect Bias finding.

**Q47. Why does a model trained on suicide prediction (D2) output "non-depressive"?**
D2 is suicide/non-suicide, not depression. "Non-depressive" is from the D1 (depression) model. These are independent predictions on different labels. The system labels make this explicit in the UI.

**Q48. XLM-RoBERTa only gets 82.4% on D3 but SVM gets 97.1% on D2 — is the transformer underperforming?**
D3 is 4-class; D2 is binary. 82.4% 4-class accuracy is equivalent to ~95%+ binary accuracy given random baselines of 25% vs 50%. XLM-RoBERTa outperforms all classical models on D3, the hardest task.

**Q49. Did you prove causality or just correlation?**
Correlation only. NLP classification finds statistical associations between text patterns and labels. No causal claims about language causing suicidality. The system is a screening signal, not a diagnostic.

**Q50. What if someone inputs non-English text?**
XLM-RoBERTa is multilingual (100 languages). Classical TF-IDF models were trained on English data — unreliable for non-English input. No language detection gate currently exists. Known limitation.

**Q51. How do you know SMOTE didn't introduce bias?**
SMOTE interpolates within the convex hull of real minority samples. Train/test split was stratified before SMOTE. Post-SMOTE validation accuracy was confirmed against pre-SMOTE baseline to check for artificial inflation.

**Q52. Why not use a single large model like GPT-4?**
(a) API cost makes real-time demo infeasible. (b) GPT-4 is a black box — no interpretability or feature importances. (c) Research goal is to study dataset-specific behavior, requiring task-specific models. (d) 278M XLM-RoBERTa is already large given our compute budget.

**Q53. Your "Ensemble Conflict" state — isn't that just saying your models disagree?**
Disagreement is informative. In clinical decision support, expressing uncertainty is safer than forcing a binary prediction. The amber state maps directly to "escalate for human review" — exactly what a responsible screening tool should do with mixed evidence.

**Q54. How would you validate this in a real clinical setting?**
Prospective validation study: present tool outputs (blinded) alongside clinician assessments on de-identified patient notes. Measure sensitivity, specificity, PPV, NPV against clinician ground truth. IRB approval required. Explicitly out of scope for this research prototype.

---

## Section 9 — Rubric Alignment Checklist

| Rubric Item | Where Addressed |
|---|---|
| Clear research question | Abstract section, RQ1/RQ2 verdict cards |
| Literature review / baseline | Related Work section, Tumaliuan baseline |
| Dataset description | D1/D2/D3 cards with class breakdown |
| Methodology / CRISP-DM | Interactive 3-step pipeline with phase tags |
| Preprocessing details | Methodology panel, SMOTE section |
| Model selection rationale | 4 architectures covering linear/margin/tree/transformer |
| Evaluation metrics | Evidence matrix + footnote re IEEE Report |
| Results presentation | Per-dataset accuracy table + confidence bars |
| Limitations acknowledged | Finding 04/05, Conclusion Lim1/Lim2 |
| Future work | Conclusion cards + MentalBERT direction |
| Live demo | Flask proxy to HuggingFace Space |
| Reproducibility | All hyperparameters documented in notebooks |
