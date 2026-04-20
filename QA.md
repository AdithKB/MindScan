# MindScan: Research Defense QA (Cross-Questions)
**Course:** NCI H9DAI · Data Analytics for Artificial Intelligence 2026  
**Objective:** Prepare for professor cross-examination based on academic requirements and project implementation.

---

## 1. Methodology & CRISP-DM
> **Professor's Focus:** Rubric requires "rigorous application" of CRISP-DM.

*   **Q: How does your 3-step dashboard map to the 6 official phases of CRISP-DM?**
    *   **A:** We mapped them as follows:
        *   **Step 1 (Data):** covers *Business Understanding* (screening objective) and *Data Understanding* (distribution analysis).
        *   **Step 2 (Preprocessing):** covers *Data Preparation* (regex cleaning, TF-IDF engineering, and SMOTE balancing).
        *   **Step 3 (Modelling):** covers *Modelling*, *Evaluation* (the Evidence Matrix), and *Deployment* (the live Flask dashboard).
*   **Q: Why choose CRISP-DM over KDD?**
    *   **A:** CRISP-DM is iterative and goal-oriented. Since our research aimed at a specific "Business Use Case" (Social Media Safety), CRISP-DM allowed us to focus on the *clinical utility* of the parallel architecture (RQ4) rather than just mathematical pattern discovery.

## 2. Data & Preprocessing
> **Professor's Focus:** Data quality, wrangling, and imbalance resolution.

*   **Q: Why use psychiatrist-verified labels (Nusrat 2024) for Dataset 1?**
    *   **A:** It ensures a high **Ground Truth** quality. Social media data often reflects self-reported "affect" (mood) rather than clinical status. Verified labels provide a standard for diagnostic accuracy across 6 clinical types.
*   **Q: Explain how SMOTE was used. Why only on the training set?**
    *   **A:** We used standard SMOTE to create synthetic minority samples in the high-dimensional TF-IDF space. We applied it **exclusively to the training set** to avoid "data leakage." The test set must represent the real-world imbalanced distribution to prove the model can generalise.
*   **Q: What specific "Knowledge Engineering" was done?**
    *   **A:** Beyond cleaning, we engineered feature vectors using **TF-IDF with bigrams** (ngram_range=1,2). This was critical for capturing negations (e.g., "not fine") which simple word-counts miss.

## 3. Modelling & Architecture
> **Professor's Focus:** Parallel vs Sequential debate (RQ4).

*   **Q: Why parallel architecture? Isn't a sequential gate more logical?**
    *   **A:** **RQ4 Answer.** A sequential gate (only check suicide if depressed) misses cases of **"Masked Suicidality."** Our research (Sample 3) shows that pre-crisis language can appear calm and resolved (Not Depressed). Parallel screening identifies these high-risk markers independently.
*   **Q: Why does SVM beat the Transformer on Dataset 1?**
    *   **A:** **RQ1 Answer.** D1 tweets average 31 words. Transformers rely on long sequence context to excel. On short, keyword-dense text, the high-dimensional SVM boundary is more effective than the transformer's attention mechanism.
*   **Q: Why did XGBoost accuracy collapse on the larger D3 splits?**
    *   **A:** **RQ2 Answer.** XGBoost showed instability at scale (91.6% at 50K → 60.1% at 116K). This suggests the model overfitted to the smaller distribution or required significant hyperparameter retuning as the feature density changed.

## 4. Evaluation & Results
> **Professor's Focus:** Accuracy metrics vs. baseline.

*   **Q: Requirement 103 states Accuracy is "not sufficient." Why is it the focus of the UI?**
    *   **A:** Accuracy was selected for the dashboard to provide an intuitive "Research Story" during the 10-minute presentation. However, full performance metrics—**Macro F1**, **Cohen's Kappa**, and **AUC-ROC**—are documented in the **IEEE Report** to satisfy technical rigor.
*   **Q: Did quadrupling the training data improve results in the D3 Split Study?**
    *   **A:** No. XLM-RoBERTa Accuracy moved only **0.1%** (98.1% vs 98.0%) from 50K to 232K samples. This proves that for transformers, **distribution and clinical relevance** (Dataset sampling) are more important than raw volume.

## 5. Limitations & Future Work
> **Professor's Focus:** Critical analysis and future direction.

*   **Q: What is the "Domain Generalization Gap" in your project?**
    *   **A:** Dataset 2 (Twitter) focuses on emotional affect. When tested on clinical clinical language (anhedonia/fatigue), the model fails. This highlights the gap between "social media sadness" and "clinical depression criteria."
*   **Q: How would you improve this in 2025?**
    *   **A:** I would move to **Multi-Task Learning (MTL)** where a single transformer backbone learns depression and suicide signals simultaneously. This reduces "Model Drift" and is significantly more efficient for real-time social media monitoring.
