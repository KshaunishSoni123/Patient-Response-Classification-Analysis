# Results analysis — final model (v4)

This analysis covers the final pipeline run (v4, Nov 24). All three models are trained on one
shared synthetic train / val / test split, and the 437 real healthcare test messages are kept
as the final external test. Every number below comes from a file in `reports/` or `models/`;
the source is named where it matters. Earlier versions (v1–v3) are compared in section 2.

## Data and splits

| Split | Messages | What it contains | Used for |
|---|---|---|---|
| synthetic train | 8,206 | rewrites of real **train** messages | training all three models |
| synthetic val | 844 | rewrites of half the real **val** messages (grouped by seed) | early stopping and seed choice, all three models |
| synthetic test | 896 | rewrites of the other half of the real val messages | per-model and **end-to-end** test |
| real test | 437 | real messages, never used as generation seeds | final external test |

The split is defined in `synthetic_splits.py` (scheme `seed-follows-real-split/v2`).
Synthetic val/test only rewrite messages that no model trained on, so the full pipeline is
scored on held-out messages without leakage.

## 1. Summary

| Model | Synthetic val | Synthetic test | Real / out-of-domain test |
|---|---|---|---|
| Intent LSTM (4x64) | 0.838 | **0.836** [0.81, 0.90], acc 0.907 | real test: **0.740** [0.66, 0.81], acc 0.886 |
| Sentiment head (frozen BERT + 2x64) | 0.949 | **0.942** [0.93, 0.96] | customer support (OOD): **0.834** [0.81, 0.86] |
| Action DNN (LSTM state, 1x64), end-to-end | 0.882 | **0.863** [0.81, 0.90], acc 0.923 | real test: **0.830** [0.74, 0.89], acc 0.918 |
| Routing table (rules), end-to-end | — | 0.857 [0.79, 0.90], acc 0.929 | real test: 0.847 [0.77, 0.90], acc 0.918 |

All scores are macro F1, with 95% bootstrap CIs in brackets (`reports/summary_metrics.csv`).
"End-to-end" means the action model receives the *predicted* intent and sentiment, not the
true labels.

The main findings:

1. **Training everything on synthetic data gives the best full pipeline.** Compared with v3
   (intent trained on real data), the action DNN gains +0.13 macro F1 on synthetic test
   (0.737 → 0.863) and +0.01 on real test (0.816 → 0.830). Intent on synthetic test goes from
   0.510 to 0.836.
2. **The price is about 3–5 points of real-test intent macro F1** (0.773 → 0.740; seed means
   0.779 → 0.730). The model learns the synthetic labeller's intent definitions and its
   emotional writing style. The real messages are short and flat, and they're labelled by
   regex rules with different definitions (section 3).
3. **Emergency recall is much better.** On synthetic test (29 emergency intents, 36 emergency
   actions), intent emergency recall rose from 0.28 to 0.76 and action DNN recall from 0.44
   to 0.75. On real test, all 3 emergencies are still caught, with 12 messages flagged.
4. **The action DNN and the routing table are tied** (0.863 vs 0.857 on synthetic test,
   0.830 vs 0.847 on real test, with heavily overlapping CIs). Neither reference is fully
   independent of the routing table, so this comparison is weak (section 5).
5. **Depth still adds little.** Linear probes on each layer score about the same as the model
   output. The 1x64 action head's input probe (0.810 on real test) is close to the trained
   head (0.830), and the intent model's four ReLU layers add nothing measurable over the LSTM
   state on synthetic data (section 6).

---

## 2. Version history

| Version | Intent trained on | Action DNN | Intent macro F1 (synth / real test) | Action DNN macro F1 (synth / real test) | Routing table (synth / real) |
|---|---|---|---|---|---|
| v1 | synthetic (old split) | avg-pooled embedding, 5x64 | — / 0.726 | — / 0.747 | — / 0.825 |
| v2 | synthetic + real train (old split) | LSTM state, 1x64 | — / 0.767 | — / 0.806 | — / 0.849 |
| v3 | real train | LSTM state, 1x64 | 0.510 / 0.773 | 0.737 / 0.816 | 0.657 / 0.859 |
| **v4 (final)** | **synthetic train** | **LSTM state, 1x64** | **0.836 / 0.740** | **0.863 / 0.830** | **0.857 / 0.847** |

v1 and v2 used an earlier synthetic split, so their synthetic scores aren't comparable and
are left out. v1's in-distribution numbers were 0.854 (intent) and 0.909 (action).
The v2 and v3 models are kept in `models/archive/both/` and `models/archive/real/`.

The style gap runs both ways. Intent trained on real data scores 0.51 on synthetic rewrites,
and intent trained on synthetic data loses a few points on real messages. v4 was chosen
because it gives the best end-to-end and emergency results, and both test sets show the
same direction for the action model.

---

## 3. Intent model

### Training

- **Hyperparameters:** the best trial of the real-data Optuna study (trial 25, CV macro F1
  0.780): Word2Vec 200-d initialised and fine-tuned, unidirectional LSTM 128, 4x64 ReLU,
  dropout 0, lr 7.9e-4, batch 32 (`models/intent_lstm_tuning/`). v3 used the same
  architecture, so v3 vs v4 differs only in the training data.
- **Training:** synthetic train, early stopping on synthetic val. The best of 3 seeds was
  chosen on synthetic val (seed 123, epoch 8).
- **Seeds on real test:** accuracy 0.878 ± 0.009, macro F1 0.730 ± 0.007. That spread is much
  smaller than v3's (± 0.016).
- **Overfitting:** synthetic train macro F1 0.991 vs val 0.838 and test 0.836. The rewrites of
  one seed share a lot of wording, so the model memorises training seeds. Val and test agree,
  so the held-out estimate is reliable.

### Synthetic test (896 messages)

Most classes score 0.89–1.00. The weak ones are:

| Class | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| general_health_info | 34 | 0.71 | **0.29** | 0.42 |
| emergency | 29 | 0.71 | 0.76 | 0.73 |
| payment | 24 | 1.00 | 0.71 | 0.83 |
| provide_information | 37 | 0.76 | 0.95 | 0.84 |

general_health_info is the vaguest category, and most of its misses go to more specific
intents such as symptom_check and medication_info.

### Real test (437 messages)

Strong classes: book_appointment (F1 0.97), symptom_check (0.96), reschedule (1.00),
cancel (1.00), referral (1.00), prescription_refill (0.90), medication_info (0.89).

Weak classes (`reports/per_class_metrics.csv`, `reports/confusion_matrices/intent_lstm__test__real.csv`):

| Class | Support | Precision | Recall | What happens |
|---|---|---|---|---|
| insurance | 93 | 1.00 | **0.76** | 11 → provide_information, 7 → account_update |
| account_update | 3 | **0.27** | 1.00 | absorbs 7 insurance "update my details" messages |
| provide_information | 10 | **0.38** | 0.80 | absorbs 11 "my policy number is …" messages |
| general_health_info | 6 | 0.38 | 0.50 | scattered confusions in both directions |
| emergency | 3 | 0.33 | 1.00 | 6 symptom_check messages flagged as emergency |

**Most of these are label-definition mismatches, not model failures.** Real labels come from
`INTENT_RULES` applied to the dataset's raw intent names. Synthetic labels come from Claude
reading the text. The two disagree in systematic ways:

- *"My policy number is XYZ789012."* is labelled insurance in the real data. The model says
  provide_information, which matches what the message is doing.
- *"My insurance provider changed. How do I update my information?"* is labelled insurance.
  The model says account_update.
- *"I'm experiencing sharp chest pains."* is labelled symptom_check. The model says
  emergency.

The synthetic labels show the same drift. Of the rewrites seeded from real *insurance*
messages, the blind labeller re-labelled 51 as account_update and 39 as provide_information.
Rewrites of symptom_check seeds were re-labelled as emergency 28 times. Trained only on
synthetic data, v4 follows the labeller's definitions more closely than v3 did. That is why
its insurance recall dropped (v3 learned the regex definitions from real train).

---

## 4. Sentiment model

| Evaluation | Accuracy | Macro F1 |
|---|---|---|
| 5-fold seed-grouped CV (synthetic train + val) | 0.940 ± 0.005 | 0.940 ± 0.005 |
| Baseline: LR on the frozen model's own 3 outputs | 0.697 ± 0.011 | 0.695 ± 0.012 |
| Synthetic val | 0.949 | 0.949 |
| Synthetic test (held-out seeds) | 0.942 | 0.942 |
| Customer support (never trained on) | 0.816 | 0.834 |

- **The trained head is worth it.** It beats a logistic regression on the frozen
  Hospital_Reviews model's original Negative/Mixed/Positive outputs by about 25 points of macro
  F1. The pretrained label space doesn't map cleanly onto patient-message tone.
- **The classes are balanced and clean in-domain.** Per-class F1 on synthetic test is
  0.93–0.95, and neutral is the hardest class. The 3 seeds agree closely (test 0.941 ± 0.002).
- **Out of domain, it under-calls negative.** On customer support, negative recall is only
  0.60 (the misses become neutral) and neutral precision is 0.67. Mild complaints are read as
  neutral. This is down from 0.900 in v1, which trained on all synthetic seeds; the
  shared split trains only on the rewrites of real *train* seeds.
- **Real healthcare messages are almost all neutral.** Of the 2,907 messages, the head labels
  2,633 (91%) neutral, 200 (7%) negative and 74 (2.5%) positive. This reflects the source
  dataset (mostly terse requests), and it means positive-sentiment actions
  (thank_you_message) are rare in the real test set (7 messages).
- **Hand labels:** 25 of the 437 real test messages are labelled so far, all neutral, and the
  head agrees on all 25. That is too few to estimate real-data sentiment accuracy
  (`labels/`, `reports/human_labels_info.json`).

---

## 5. Action DNN

### Training

- **Architecture:** input = intent LSTM state (128) + intent probabilities (17) + sentiment
  probabilities (3), then 1x64 ReLU, then softmax over 8 actions (chosen in section 7's
  experiments).
- **Inputs:** synthetic train rows use out-of-fold intent and sentiment predictions (5
  seed-grouped folds), so the DNN learns from realistically noisy inputs. Val/test rows use
  the final models' predictions, which makes the synthetic val/test scores end-to-end.
- **Seeds:** synthetic test macro F1 0.857–0.863 over 3 seeds; early stopping picked epochs
  5–7. Seed 7 was chosen on synthetic val (0.882).

### DNN vs routing table

Per-class F1 on synthetic test (end-to-end) and real test:

| Action | Synth support | DNN P / R / F1 (synth) | Table P / R / F1 (synth) | Real support | DNN F1 (real) | Table F1 (real) |
|---|---|---|---|---|---|---|
| route_to_human | 565 | 0.97 / 0.93 / 0.95 | 0.96 / 0.95 / 0.96 | 220 | 0.95 | 0.95 |
| get_practice_info | 82 | 0.84 / 0.93 / 0.88 | 0.87 / 0.95 / 0.91 | 25 | 0.73 | 0.74 |
| book_appt | 74 | 0.88 / 0.93 / 0.91 | 0.92 / 0.91 / 0.91 | 78 | 0.98 | 0.97 |
| reschedule_appt | 50 | 0.96 / 1.00 / 0.98 | 0.96 / 0.98 / 0.97 | 20 | 0.98 | 1.00 |
| thank_you_message | 44 | 0.84 / 0.93 / 0.88 | 0.95 / 0.93 / 0.94 | 7 | 0.92 | 1.00 |
| send_explanation_of_benefits | 37 | 0.83 / 0.92 / 0.87 | 0.86 / 0.86 / 0.86 | 79 | 0.88 | 0.85 |
| route_to_human_emergency | 36 | 0.69 / 0.75 / **0.72** | 0.62 / 0.67 / 0.64 | 3 | 0.40 | 0.46 |
| send_payment_link | 8 | 0.67 / 0.75 / 0.71 | 1.00 / 0.50 / 0.67 | 5 | 0.80 | 0.80 |

The DNN is ahead where safety matters (emergency, +0.08 F1) and on benefits requests. The
table is ahead on thank-you messages and practice info. Everything else is level.

**How to read this comparison:** the synthetic action labels were made by applying the
routing table to the *true* intent and the LLM's sentiment, then reviewed. The real-test
reference applies the same table to the regex intent and the sentiment head's prediction.
So whenever the intent prediction is right, the table agrees with the reference almost by
construction. In v3, the DNN led on synthetic test by 0.08 because it coped better with a
weak intent model. Now that intent is good, the two are tied. Only a human-labelled action
reference could show whether the DNN's deviations are better or worse than the rules.

### Error decomposition (36 DNN errors on real test, `models/action_eval/errors.csv`)

| Cause | Errors | Notes |
|---|---|---|
| Intent model wrong → wrong action | 29 | Inherited; the routing table makes the same mistake on 27. Most are the section-3 label-definition cases: insurance → provide_information (7) / account_update (4), symptom_check → emergency (6). |
| Intent right, DNN deviates from the policy | 7 | See below |

Down from 52 errors in v1 (33 inherited, 19 deviations). The 7 deviations, read one by one:

- **3:** symptom messages escalated to emergency (*"I'm experiencing severe headache and
  dizziness"* ×2, *"sharp chest pains"*). These are clinically reasonable escalations; the
  reference says route_to_human only because the source labels them symptom_check.
- **1:** an insurance message the sentiment head marked *positive* (*"My insurance policy
  doesn't seem to cover this procedure."*). The reference says thank_you_message, and the
  DNN's explanation of benefits is more sensible. Here the reference is wrong.
- **1:** a mild insurance question marked negative (*"My insurance provider is changing next
  month…"*) → DNN sends explanation of benefits, while the policy routes negatives to a human.
  This is a sentiment-definition difference.
- **1:** *"I need to reschedule my cardiology appointment…"* (sentiment: negative) → DNN
  reschedules, while the policy routes to a human. That is arguably the better action.
- **1 genuine error:** *"It started about three days ago, around March 14th."* (a follow-up
  detail) → route_to_human instead of get_practice_info.

So only 1–2 of the DNN's own decisions are clearly wrong. The rest are label-definition
differences or cautious escalations.

### Emergencies (`reports/emergency_metrics.csv`)

| | Synthetic test recall | Synthetic test precision | Real test |
|---|---|---|---|
| Intent model (emergency intent) | 0.76 (22/29) | 0.71 | 3/3, 9 flagged |
| Action DNN | **0.75** (27/36) | 0.69 | 3/3, 12 flagged |
| Routing table | 0.67 (24/36) | 0.62 | 3/3, 10 flagged |

The synthetic test is the first set with enough emergencies (36) to measure recall: Wilson
95% CI for the DNN [0.59, 0.86]. One emergency in four is still missed. On real test, most
of the extra flags are chest pain / dizziness / shortness-of-breath messages labelled
symptom_check, which is the safer direction to err in.

---

## 6. Hidden-layer analysis (`reports/hidden_layers/`)

**Linear probes** (logistic regression on each layer's activations, macro F1):

| Model | Input | First layer | Last hidden | Model output |
|---|---|---|---|---|
| Intent, synthetic test | 0.836 (avg Word2Vec) | 0.846 (LSTM) | 0.844 (hidden_4) | 0.836 |
| Intent, real test | 0.735 | 0.719 (LSTM) | 0.744 | 0.740 |
| Sentiment, synthetic test | 0.944 (BERT CLS) | 0.943 | 0.946 | 0.942 |
| Sentiment, customer support | 0.850 | 0.887 | 0.872 | 0.834 |
| Action, synthetic test | 0.858 | 0.862 (hidden_1) | 0.862 | 0.863 |
| Action, real test | 0.810 | 0.802 | 0.802 | **0.830** |

- **Intent:** the classes are already about as separable as they get at the LSTM state. The
  four ReLU layers keep the information but don't add to it. On real test, hidden_1 is
  slightly better than the LSTM state (0.757 vs 0.719), but the later layers fall back.
- **Sentiment:** the 2x64 head adds value out of domain at its first layer (0.850 → 0.887).
  The final softmax scores below the hidden-layer probes (0.834), so the output layer is
  tuned to synthetic tone.
- **Action:** unlike v1 (input probe 0.787 > DNN 0.747), the trained 1x64 head now *beats* a
  linear model on its inputs on real test (0.830 vs 0.810). That supports the switch to a
  shallow head.

**Dead units grow with depth.** In the intent model, 0% of hidden_1's units are inactive,
4.7% of hidden_2's, 15.6% of hidden_3's and 14.1% of hidden_4's. The action DNN (1.6%) and
the sentiment head (0–6%) have almost none.

**Specialist nodes are real but redundant.** Zeroing a class's top specialist nodes lowers
that class's recall by 2.5 points (intent), 2.5 (action) and 0.6 (sentiment), against
−0.2 to +0.1 for random nodes. Other classes are unaffected (+0.1 to +0.2). The
specialist effect beats the random control's 5th percentile in 46 of 170 intent ablations, 4
of 16 action ablations and 3 of 12 sentiment ablations. Classes are encoded by identifiable
units, but with enough redundancy that removing a few barely matters. The sentiment head is
the extreme case: 54–60 of its 64 units are specialists for one of only 3 classes.

---

## 7. Action-architecture experiments (`reports/experiments/action_architecture/`)

These experiments chose v4's action architecture. They were run on Nov 23 with the v1/v2
intent models and the earlier split, so their absolute numbers aren't comparable with
sections 1–6; only the relative effects matter.

The grid was 2 intent sources × 2 text inputs × 10 head sizes × 3 seeds = 120 runs. Averaged
main effects on real test macro F1:

| Factor | Levels | Real test macro F1 |
|---|---|---|
| Text input | avg-pooled embedding / **LSTM state** | 0.753 / **0.782** (+0.029) |
| Intent source | synthetic / **both** | 0.763 / **0.773** (+0.010) |
| Hidden layers | **1** / 2 / 3 / 5 | **0.771** / 0.766 / 0.768 / 0.759 |
| Units | 32 / **50** / 64 | 0.763 / **0.771** / 0.769 |

- **The LSTM state is the biggest lever.** It is also by far the best at catching
  emergencies in synthetic data (emergency recall 0.93 vs 0.74), and the model is 30x smaller
  (6k vs 180k parameters, since there is no trainable embedding).
- **Shallow heads are at least as good.** 5x64 is the worst depth on both real val and real test.
- **The best configuration by real val** was `both | lstm_state | 1x64`: real val 0.854 and
  real test 0.806 ± 0.008, significantly better than v1 (+0.062, CI [0.003, 0.157]). v4 keeps
  the LSTM-state input and the 1x64 head.
- **Selecting on val is weak.** Across configurations, real-val and real-test macro F1
  correlate at only r = 0.29 (synthetic val vs real test: r = 0.37). Treat small ranking
  differences as noise; the large effects (text input, depth) are consistent on every split.

---

## 8. Conclusions and recommendations

1. **v4 is the final model.** It has the best end-to-end results on synthetic test (0.863),
   matches the routing table on real test, and has much better emergency recall than any
   earlier version.
2. **Finish the human-labelled real test set.** Only 25 of 437 messages are labelled
   (sentiment only). Both current references depend on the routing table, so they can't
   show whether the DNN's deviations (escalating chest pain, rescheduling a mildly negative
   request) are improvements. Human sentiment, and ideally action, labels are the
   highest-value next step.
3. **Align the intent label definitions.** Either re-map `INTENT_RULES` (insurance-detail
   updates → account_update, ID/policy numbers → provide_information, acute chest pain /
   breathing → emergency) or re-label the real data with the same blind labeller. 18 of the
   22 real-test insurance errors and all 6 false emergency flags come from this mismatch.
4. **Keep a rule-based emergency override on top of the DNN.** Synthetic-test recall is 0.75,
   so one emergency in four is still missed, and a missed emergency costs far more than a
   false escalation. A lower emergency threshold, chosen on synthetic val, is worth trying.
5. **Review the 1,080 synthetic rows flagged `needs_review`,** especially the insurance →
   account_update / provide_information and symptom → emergency re-labels. Since v4 trains
   only on synthetic data, these labels define what all three models learn.
6. **For the write-up,** report accuracy and weighted F1 alongside macro F1 on real test.
   Three of the eight actions have ≤ 7 real examples and emergency has 3, so macro F1 swings
   on single messages. The synthetic test (896 messages, 36 emergencies) is the steadier
   comparison.

## Figures

- `reports/confusion_matrices/*.png`: confusion matrices for every model and split
- `reports/hidden_layers/linear_probe.png`, `reports/hidden_layers/heatmaps/`: probes and
  per-node class heatmaps (primary = synthetic test, secondary = real test / customer support)
- `reports/experiments/action_architecture/real_val_macro_f1.png`: architecture grid
