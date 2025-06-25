---
hide:
- toc
---
# Dataset Description Avoidance Learning Task

This dataset contains trial-level data from an avoidance learning task. This data was first collected using a task repo and then processed using the data extraction repo (see project pipeline on [home](../index.md)). So, these data are the output of the data extraction repo. Each row corresponds to a single trial completed by a participant. The columns include identifiers, stimulus attributes, trial responses, contextual variables, and subject-level measures. All participants are included in this file; however, you are able to upload multiple files, which will be concatenated (see [parmaters](parameters.md)).

## Sample Data

## Sample Data

| participant_id         | trial_type     | rt   | symbol_L_name | symbol_R_name | feedback_L | feedback_R | context_val | choice_made | context_val_name | duration        | group_code | intensity | unpleasant | interference | age | sex    |
|------------------------|----------------|------|----------------|----------------|------------|------------|-------------|--------------|-------------------|------------------|------------|-----------|-------------|---------------|-----|--------|
| 6644c218e860482bd1d5113a | learning-trials | 1529 | 75P1           | 25P1           | -10        | 0          | -1          | 1            | Loss Avoid        | I am not in pain | 0          | 4         | 3           | 3             | 37  | Female |
| 6644c218e860482bd1d5113a | probe           | 752  | 25R2           | Zero           |            |            | 1           | 0            | Reward            | I am not in pain | 0          | 4         | 3           | 3             | 37  | Female |
| 5f27835438b9f85a6b9ebdb4 | learning-trials | 1176 | 25R1           | 75R1           | 0          | 0          | 1           | 0            | Reward            | > 5 years        | 2          | 39.5      | 38.5        | 36            | 42  | Female |

## Column Descriptions

| Column Name        | Description |
|--------------------|-------------|
| `participant_id`   | Unique identifier for each participant. |
| `trial_type`       | Type of trial: `"learning-trials"` for the learning phase and `"probe"` for the transfer phase. |
| `rt`               | Reaction time in milliseconds. |
| `symbol_L_name`    | Identifier for the symbol presented on the left. May be 75R1, 75R2, 25R1, 25R2, 25P1, 25P2, 75P1, 75P2, Zero (neutral in transfer phase). |
| `symbol_R_name`    | Identifier for the symbol presented on the right. May be 75R1, 75R2, 25R1, 25R2, 25P1, 25P2, 75P1, 75P2, Zero (neutral in transfer phase). |
| `feedback_L`       | Feedback value associated with the left symbol. May be -10, 0, or 10. This will be empty for the transfer phase |
| `feedback_R`       | Feedback value associated with the right symbol. May be -10, 0, or 10. This will be empty for the transfer phase |
| `context_val`      | Encoded task context value: typically `1` (reward), `-1` (punishment), or `0` (neutral; only in transfer phase). |
| `choice_made`      | Binary indicator for the participant's choice: `1` for right, `0` for left. |
| `context_val_name` | Human-readable description of `context_val`, e.g., `"Reward"` or `"Loss Avoid"`. |
| `duration`         | Self-report string describing current pain state, e.g., `"I am not in pain"`. |
| `group_code`       | Participant group code (typically categorical or condition label): `0`: no pain group, `1`: acute pain group, `2`: chronic pain group. |
| `intensity`        | Self-reported pain intensity rating. |
| `unpleasant`       | Self-reported unpleasantness rating. |
| `interference`     | Self-reported rating of pain interference with daily life. |
| `age`              | Participant's age in years. |
| `sex`              | Participant’s reported sex (e.g., `"Female"`, `"Male"`, `"Other"`). |
