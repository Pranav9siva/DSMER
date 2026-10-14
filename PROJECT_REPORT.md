# Multimodal Speech Emotion Recognition Project Report

> **Status:** Report draft based on the repository's current implementation and saved evaluation artifacts. Fill in the author, course, faculty, and date fields before submission.

## Project Information

- **Project title:** Multimodal Speech Emotion Recognition and Acoustic Analysis
- **Student/team:** [Add name(s)]
- **Faculty advisor:** [Add name]
- **Course/program:** [Add course or program]
- **Report date:** [Add date]
 **Report date:** [Add date]

## Team Contributions

The following is a suggested four-member division of work. Replace the placeholders with team members' names and adjust the descriptions to reflect the work each person actually completed.

| Team member | Primary contribution | Example responsibilities and deliverables |
|---|---|---|
| **Member 1: [Name]** | Dataset and model development | Prepared and inspected MELD inputs; reviewed class distribution; configured and trained the HMEMNet model with the DistilBERT text backbone; tracked development metrics and checkpoint selection. |
| **Member 2: [Name]** | Backend and media processing | Implemented or maintained FastAPI inference endpoints; integrated model loading, audio/video decoding, transcription, and speaker-aware turn handling; added librosa F0 extraction to prediction responses. |
| **Member 3: [Name]** | Frontend and user experience | Built the browser-based input and prediction flow; displayed emotion labels and confidence; added the interactive pitch chart and F0 summary values; checked responsive presentation. |
| **Member 4: [Name]** | Evaluation, testing, and report | Ran or reviewed evaluation and regression checks; summarized accuracy, F1 scores, per-class results, and confusion patterns; documented limitations, reproducibility steps, and future experiments. |

All members should review the integrated system and final report. This table is a role-allocation template, not evidence of individual authorship; report only contributions confirmed by the team.

## Abstract

This project develops a speech emotion recognition prototype for conversational data from the Multimodal EmotionLines Dataset (MELD). The system classifies utterances into seven emotion categories and provides a browser-based interface for transcript and media inputs. Its inference pipeline uses an HMEMNet-based model with a fine-tuned DistilBERT text backbone and supports audio/video upload workflows, transcription, and speaker-aware turn handling. As an additional interpretability feature, the application extracts fundamental frequency (F0) from uploaded audio or video audio using librosa and displays the pitch contour and summary statistics alongside the predicted emotion and confidence.

The saved MELD test evaluation contains 2,610 examples and reports 59.35% accuracy, 58.66% weighted F1, and 40.16% macro-F1. The F0 extraction and chart have been verified as application functionality. They have **not** been evaluated as an input to the trained classifier, so this project does not claim that F0 improves emotion-classification metrics.

## Executive Summary

The deliverable is an end-to-end research prototype: a trained seven-class classifier, a FastAPI inference service, a browser interface for text/audio/video, transcription and speaker-aware turn processing, and an auxiliary F0 analysis visualization. The evaluated model reached 59.35% accuracy on the saved MELD test evaluation. Its 40.16% macro-F1 is notably lower than its 58.66% weighted F1, indicating uneven performance across classes rather than uniformly strong recognition.

The software work and the research claim should be distinguished. The pitch feature is implemented and its extraction/visualization path was smoke-tested. The classifier was not retrained with F0, and no controlled F0 ablation exists. Therefore, this report presents F0 as an interpretability/analysis feature and identifies a controlled ablation as future work.

## 1. Introduction

Emotion in conversation is expressed through both language and vocal delivery. A transcript can convey lexical and contextual cues, while speech acoustics such as pitch can provide complementary descriptive information. This project combines a trained emotion-classification system with a usable inference interface and adds a separate pitch-analysis view to make one acoustic property inspectable.

### Objectives

1. Train and evaluate a seven-class emotion classifier using MELD.
2. Package the selected model and text backbone for local inference.
3. Provide an interface for transcript, audio, and video inputs.
4. Display predicted emotion, confidence, and available conversation-turn information.
5. Extract and visualize F0 for uploaded media as an auxiliary acoustic analysis.

### Scope and deliverables

- HMEMNet training and checkpoint selection using MELD training/development splits.
- Exported TorchScript model and matching text-backbone artifacts for inference.
- FastAPI `/predict` and `/health` endpoints with static browser UI.
- Audio/video decoding and best-effort transcription; speaker-aware turn construction with fallback segmentation.
- Librosa-based F0 estimation and an interactive pitch chart. This feature is not a second trained classifier and is not included in the reported model metrics.

## 2. Dataset and Experimental Setup

The recorded training run used MELD with 9,989 training examples and 1,109 development examples. The saved test evaluation contains 2,610 examples. The seven target classes are anger, disgust, fear, joy, neutral, sadness, and surprise.

The training configuration records a fine-tuned DistilBERT text backbone, batch size 1, seed 42, and CUDA as the training device. The best checkpoint was selected using development macro-F1. The training history reports best development macro-F1 at epoch 9; the exported artifacts are listed in the training metadata.

### Training class distribution

The training metadata records the following class counts. The imbalance is substantial: neutral is the largest class, while fear and disgust are much smaller.

| Emotion | Training examples | Share of training set |
|---|---:|---:|
| Anger | 1,109 | 11.1% |
| Disgust | 271 | 2.7% |
| Fear | 268 | 2.7% |
| Joy | 1,743 | 17.4% |
| Neutral | 4,710 | 47.2% |
| Sadness | 683 | 6.8% |
| Surprise | 1,205 | 12.1% |

Training code uses inverse-square-root-frequency weighted sampling to reduce the effect of class imbalance. This changes sampling exposure; it does not make the original dataset balanced.

### Recorded training configuration

| Setting | Recorded value |
|---|---|
| Text backbone | `distilbert-base-uncased`, fine-tuning enabled |
| Hidden representation size | 256 |
| Batch size | 1 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Maximum requested epochs | 20 |
| Early-stopping patience | 5 epochs without development macro-F1 improvement |
| Text-backbone unfreeze epoch | 2 |
| Random seed | 42 |
| Training device in metadata | CUDA |
| Checkpoint selection criterion | Development macro-F1 |

The recorded run stopped after 14 epochs. Its best development macro-F1 occurred at epoch 9 (0.4097); the training summary's `best_epoch` field refers to a different best-loss criterion and should not be confused with the best development macro-F1 checkpoint.

## 3. System and Methods

### Emotion classification

The model implementation is HMEMNet (Hierarchical Multi-scale Emotional Memory Network). It contains modality-specific audio, text, and vision encoders; projects and fuses their representations; applies temporal transformer processing; and uses person, dialogue, and conversation memories with a memory controller and memory-guided fusion. The final emotion head produces seven class logits over temporal features. The configured temporal transformer uses four attention heads. Speaker IDs are included in the memory pathway.

The application loads the exported TorchScript model and matching text-backbone artifacts. The API accepts transcript text and uploaded audio/video. For media inputs it attempts speech transcription, constructs utterance/turn inputs, and uses available diarization or an energy-based fallback to create speaker-aware segments. The response includes the predicted emotion label and confidence, transcript provenance, input-modality metadata, and any segment-level results. When a media upload is present, the F0 summary is also returned.

### Inference workflow

1. The browser submits transcript text and/or an audio/video file to `/predict`.
2. The backend reads the upload and records which modalities were provided.
3. For media, it attempts transcription; if that fails and there is no supplied transcript, a fallback transcript label is used.
4. The backend builds turns using available diarization or energy-based segmentation and prepares model inputs.
5. HMEMNet produces class logits; the API returns the highest-confidence emotion and confidence score, plus turn details where applicable.
6. Independently of classification, uploaded media is decoded for F0 estimation; the frontend displays the resulting pitch contour and summary statistics when a usable contour is available.

The separate F0 branch is important: drawing a pitch curve for an inference does not mean pitch affected that inference's class prediction.

### F0 extraction and visualization

For uploaded audio or the audio track of uploaded video, the API decodes a mono waveform, resamples it to 16 kHz, and estimates the F0 contour using `librosa.yin`. The response includes time and frequency samples plus mean, minimum, and maximum F0. The frontend renders these values as an interactive pitch graph in the prediction panel.

F0 currently serves as a post-inference analysis and visualization feature. It is not a demonstrated model input in the saved training/evaluation run.

The pitch estimator is configured for a 50-500 Hz search range. Unvoiced or unusable estimates are omitted from the plotted contour. If the optional librosa import or pitch estimation fails, the API preserves prediction behavior and returns an empty pitch summary rather than failing the emotion request.

### Evaluation protocol

The reported test scores are read from the saved `outputs/evaluation_test.json` classification report for the test split (2,610 examples). Accuracy is the proportion of correct predictions. Per-class precision and recall are reported in the artifact, and per-class F1 is their harmonic mean. Weighted F1 weights each class by its support; macro-F1 averages class F1 values equally and therefore better exposes weak performance on less frequent classes. Checkpoint selection used development macro-F1, not test performance.

## 4. Results

### Test-set performance

Results below are taken from `outputs/evaluation_test.json` and correspond to 2,610 test examples.

| Metric | Score |
|---|---:|
| Accuracy | 59.35% |
| Weighted F1 | 58.66% |
| Macro-F1 | 40.16% |

### Per-class test results

| Emotion | Precision | Recall | F1 |
|---|---:|---:|---:|
| Anger | 0.412 | 0.365 | 0.387 |
| Disgust | 0.207 | 0.279 | 0.238 |
| Fear | 0.091 | 0.080 | 0.085 |
| Joy | 0.592 | 0.520 | 0.554 |
| Neutral | 0.731 | 0.791 | 0.759 |
| Sadness | 0.338 | 0.250 | 0.287 |
| Surprise | 0.483 | 0.520 | 0.501 |

Neutral was the strongest class by F1 (0.759), while fear was the weakest (0.085). This gap, along with the relatively low macro-F1, indicates that performance is uneven across emotion classes and that minority/emotionally ambiguous classes remain challenging.

### Error-pattern analysis

The saved confusion matrix uses the class order shown in the per-class table. A few notable test-set patterns are:

- Of 50 fear examples, 4 were classified correctly; 17 were predicted as neutral. Fear recall was 0.080.
- Of 208 sadness examples, 52 were classified correctly; 85 were predicted as neutral. Sadness recall was 0.250.
- Of 402 joy examples, 209 were classified correctly; 103 were predicted as neutral. Joy recall was 0.520.
- Of 1,256 neutral examples, 993 were classified correctly, consistent with the neutral recall of 0.791.

These patterns show a tendency for several non-neutral emotions to be absorbed into the neutral class. They motivate closer review of context, vocal expression, class imbalance, and labeling ambiguity. They do **not** show that pitch would resolve those errors; that requires a controlled, labeled comparison.

### Development checkpoint selection

The saved training history records a best development macro-F1 of 0.4097 at epoch 9. This is a development-set checkpoint-selection value, not a test result. The final test metrics above should be used when reporting held-out performance.

### F0 feature verification and examples

A generated 220 Hz sine-wave upload was used as an engineering smoke test. The live endpoint returned 32 pitch samples and a mean estimate of 220.23 Hz. This validates that media decoding, pitch extraction, and response serialization work for that test signal. It is a synthetic signal, not a labeled emotional utterance, and it does not show that F0 improves emotion classification.

No verified, labeled examples currently demonstrate that the pitch feature corrected a model prediction. To make that claim, compare the same test utterances with and without F0 in a controlled model ablation, and inspect examples where the predicted class changes correctly against the label.

### Software verification

The focused pipeline test file was run after the F0 response was added: 9 tests passed. The suite exercises transcript handling, turn construction/diarization behavior, and prediction response behavior. A live API smoke test using a generated 220 Hz sine wave returned an emotion response, 32 pitch values, and mean F0 of 220.23 Hz. The synthetic tone is useful for checking pitch estimation accuracy and response wiring, but it is not a human emotion-recognition example. This was a focused pipeline check, not evidence that every test in the repository has passed.

## 5. Discussion

The test results show moderate overall accuracy, but macro-F1 is substantially lower than weighted F1 because class-level performance varies. The 18.50 percentage-point difference between weighted F1 (58.66%) and macro-F1 (40.16%) is consistent with the uneven per-class scores: the larger neutral class performs relatively well, while fear performs poorly. The confusion matrix also shows that neutral is a common destination for errors from fear, sadness, and joy.

The F0 graph adds an interpretable acoustic view to the application and offers a path for future research. It can help a user inspect pitch patterns that may be relevant to prosody, but it currently does not explain or alter the classifier's output. Pitch alone is not an emotion label: speaker identity, language, context, loudness, speaking style, and recording quality can all affect F0.

The current implementation is therefore best described as an emotion-classification prototype with supplementary pitch analysis, rather than a validated F0-enhanced classifier.

## 6. Limitations

- Performance is based on the saved MELD test evaluation and may not generalize to other domains, speakers, recording conditions, or languages.
- Class performance is imbalanced; fear has low precision, recall, and F1 in the reported test results.
- No controlled F0-enabled versus F0-disabled model ablation has been run.
- The pitch smoke test uses a synthetic tone and does not establish emotion-recognition value.
- Automatic transcription and speaker segmentation depend on external tools and can fail or produce imperfect results on noisy media.
- No confidence intervals or multi-seed variability estimates are recorded in the current evaluation artifact.
- The saved report gives aggregate and class-level metrics, but no human-reviewed examples or error annotations are included.
- A single best checkpoint/run does not establish statistical significance or robustness to alternate train/dev/test partitions.
- F0 estimates can be unreliable for unvoiced speech, overlapping speakers, noisy recordings, and voices outside the configured frequency search range.
- The web application is a local prototype; deployment security, concurrency, and production-scale operational behavior have not been established by the reported checks.

## 7. Future Work

1. Integrate acoustic F0 features into the training data/model in a defined, reproducible way. Candidate features include median F0, range, variability, slope, and normalized pitch contours; compare feature choices rather than assuming a single statistic is sufficient.
2. Run a controlled baseline-versus-F0 ablation using identical splits, seeds, optimizer settings, checkpoint criteria, and evaluation procedures. Avoid using the test set for feature selection.
3. Report accuracy, macro-F1, weighted F1, per-class precision/recall/F1, confusion matrices, and confidence intervals across multiple seeds where feasible.
4. Review labeled examples from fear, sadness, joy, and their neutral confusions. Record the gold label, baseline prediction, F0-enabled prediction, transcript, and relevant pitch statistics; include only anonymized examples permitted by dataset terms.
5. Address low-performing classes using documented strategies such as class-aware sampling, calibration, or targeted data analysis, and verify any gain on held-out data.
6. Evaluate robustness across speakers, recording conditions, and media types, and quantify ASR/diarization failures separately from classifier errors.
7. Improve operational documentation and add API-level tests for media decoding, empty/failed F0 extraction, and response schema compatibility.

### Suggested ablation table for the next study

| System | Model inputs | Test accuracy | Weighted F1 | Macro-F1 | Fear F1 | Sadness F1 | Surprise F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | Existing trained inputs; no explicit F0 feature | 0.5935 | 0.5866 | 0.4016 | 0.0851 | 0.2873 | 0.5009 |
| F0-enhanced | Same inputs plus defined F0 features | [Run experiment] | [Run experiment] | [Run experiment] | [Run experiment] | [Run experiment] | [Run experiment] |

The baseline values above are copied from the saved evaluation artifact. Complete the second row only after training and evaluating a matched F0-enabled model. If resources allow, repeat both systems across multiple seeds and report mean and standard deviation.

## 8. Conclusion

The project delivers a working MELD-based emotion-recognition application, a saved held-out test evaluation, and an interactive F0 analysis feature for uploaded media. The evaluated classifier achieved 59.35% accuracy and 40.16% macro-F1 on 2,610 test examples. The pitch analysis successfully extracts and visualizes F0, but further controlled experiments are required before claiming that it improves classification performance.

## Reproducibility References

- Project setup, training, evaluation, and run instructions: [`README.md`](README.md)
- Training history and selected development checkpoint: [`outputs/training_results.json`](outputs/training_results.json)
- Held-out test metrics and confusion matrix: [`outputs/evaluation_test.json`](outputs/evaluation_test.json)
- Development metrics artifact: [`outputs/evaluation_dev.json`](outputs/evaluation_dev.json)
- Backend API and F0 extraction: [`DSMER/deployment/backend/api/app.py`](DSMER/deployment/backend/api/app.py)
- Frontend prediction and pitch visualization: [`DSMER/deployment/frontend/index.html`](DSMER/deployment/frontend/index.html)

## Appendix A: Reproducing the Recorded Workflow

Run these commands from the repository root in PowerShell. The dataset media and FFmpeg must be available as described in the project README.

```powershell
# Install project dependencies
.\.venv\Scripts\python.exe -m pip install -r DSMER\requirements.txt

# Train (the training script selects a checkpoint using development macro-F1)
.\.venv\Scripts\python.exe DSMER\train.py --media_root dataset\MELD\media --batch_size 1

# Evaluate the selected checkpoint on the MELD test split
.\.venv\Scripts\python.exe DSMER\tools\evaluate.py `
	--ckpt outputs\checkpoint_best.pth `
	--split test `
	--batch_size 1 `
	--media_root dataset\MELD\media `
	--text_backbone_path outputs\exports\text_backbone `
	--output outputs\evaluation_test.json

# Run the focused pipeline tests
.\.venv\Scripts\python.exe -m pytest DSMER\tests\test_multispeaker_pipeline.py -q

# Start the local inference application
.\.venv\Scripts\python.exe DSMER\deployment\backend\api\app.py
```

For a final submission, record the exact checkpoint path, evaluation command, software versions, date of evaluation, and any deviations from the saved configuration. Regenerate the test evaluation only from the selected checkpoint and matching fine-tuned text backbone; do not report development metrics as test results.

## Appendix B: Artifact Map

| Artifact | Purpose |
|---|---|
| `outputs/training_results.json` | Training history, metadata, class counts, and development checkpoint-selection metrics |
| `outputs/checkpoint_best.pth` | Best checkpoint selected by development macro-F1 |
| `outputs/exports/hmemnet_ts.pt` | TorchScript inference model |
| `outputs/exports/text_backbone/` | Matching fine-tuned text-backbone artifact |
| `outputs/evaluation_test.json` | Test metrics, class report, and confusion matrix |
| `outputs/evaluation_dev.json` | Saved development evaluation artifact |
| `DSMER/deployment/backend/api/app.py` | API, media processing, inference, and F0 extraction |
| `DSMER/deployment/frontend/index.html` | Browser UI and pitch visualization |
