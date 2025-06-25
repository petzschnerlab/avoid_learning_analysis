---
hide:
- toc
---
# Pipeline Parameters

The **Avoid Learning Analysis** package is a tool for analyzing data from the SOMA avoidance learning study. It is designed to run with customizable parameters and generates a report summarizing the analysis results.

This package can be run as a standalone tool or alongside the companion package [`avoid_learning_rl_models`](https://github.com/your-org/avoid_learning_rl_models). When used together, the `load_models` parameter enables the inclusion of model outputs in the report.

## Quick Start

```python
from helpers.pipeline import Pipeline

params = {
    'file_path': 'path/to/data',
    'file_name': ['subfolder1/data1.csv', 'subfolder2/data2.csv']
}

pipeline = Pipeline()
pipeline.run(**params)
```

!!! note
    The docstrings for this package are primarily AI-generated. The help function below is written by hand and should be considered authoritative in the case of any discrepancy.

---

# Parameters

## `Pipeline` Constructor Parameters

```python
from helpers.pipeline import Pipeline

pipeline = Pipeline(**params)
```

<table>
  <tr>
    <th><strong>NAME</strong></th>
    <th><strong>TYPE</strong></th>
    <th><strong>DEFAULT</strong></th>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>help</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Prints the help information for the package, including an overview and parameter descriptions.</td>
  </tr>
</table>

---

## `run()` Method Parameters

```python
from helpers.pipeline import Pipeline

pipeline = Pipeline()
pipeline.run(**params)
```
<table>
  <tr>
    <th><strong>NAME</strong></th>
    <th><strong>TYPE</strong></th>
    <th><strong>DEFAULT</strong></th>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>help</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Prints the help information for the package, including an overview and parameter descriptions.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>author</strong></td>
    <td><strong>str</strong></td>
    <td><strong>PEAC_Team</strong></td>
  </tr>
  <tr>
    <td colspan="3">Author of the report, which will be displayed on the report.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>print_filename</strong></td>
    <td><strong>str</strong></td>
    <td><strong>AL/reports/SOMA_report.pdf</strong></td>
  </tr>
  <tr>
    <td colspan="3">This package generates a report including some main demographics, statistics, and figures. The report will be saved at the provided location, originating from the root directory of the repo. This will overwrite any existing report with the same name, but if this report is open, it will add -N to the report name where N increases from 1 until a unique name is found.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>file_path</strong></td>
    <td><strong>str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Path to the data file(s) to be loaded. This is a required parameter. From this path, you can load several different files using the file_name parameter.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>file_name</strong></td>
    <td><strong>list[str] | str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Name of the file(s) to be loaded. This is a required parameter. These filenames should be relative to the file_path parameter. You can load multiple files by providing a list of file names or a single file name as a string. You can add further path information here if your data splits at the point of file_path. For example, file_path = "path/to/data" and file_name = ["subfolder1/data1.csv", "subfolder2/data2.csv"] will load two files from different subfolders.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>dataset</strong></td>
    <td><strong>str</strong></td>
    <td><strong>""</strong></td>
  </tr>
  <tr>
    <td colspan="3">This is a description of the dataset(s) in the report. It allows for you to provide a brief overview of the dataset(s) used in the analysis, if you like.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>split_by_group</strong></td>
    <td><strong>str</strong></td>
    <td><strong>pain</strong></td>
  </tr>
  <tr>
    <td colspan="3">Split the data by clinical group, options are 'pain' and 'depression'. Unfortunately, somewhere along the way, I stopped testing the depression group, so it crashes. A lot of functionality is in place for this group, but lots of stats/plots were added after I stopped testing it. Moreover, the loading of model data was also not tested for the depression group. So, future development can bring this back to life, potentially without much effort. But, do not use it for now.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>split_by_group_id</strong></td>
    <td><strong>str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">This ID is used as a unique identifier for the analysis. That way, you can run the pain analysis multiple times (with different parameters) without overwriting the previous results. This is designed around statistics, which saves data and results using this ID. If you do not add this parameter, it will default to the value of the split_by_group parameter.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>accuracy_exclusion_threshold</strong></td>
    <td><strong>int</strong></td>
    <td><strong>70</strong></td>
  </tr>
  <tr>
    <td colspan="3">Threshold for excluding participants based on their accuracy (%) in the learning phase. Specifically, this is the accuracy in the last quarter of the learning phase. Anyone below this threshold will be excluded from the analysis. The number of people excluded will be indicated in the report.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>RT_low_threshold</strong></td>
    <td><strong>int</strong></td>
    <td><strong>200</strong></td>
  </tr>
  <tr>
    <td colspan="3">Lower threshold for excluding trials based on reaction times (ms).</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>RT_high_threshold</strong></td>
    <td><strong>int</strong></td>
    <td><strong>5000</strong></td>
  </tr>
  <tr>
    <td colspan="3">Upper threshold for excluding trials based on reaction times (ms).</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>rscripts_path</strong></td>
    <td><strong>str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Path to the R executible file, which is used when running GLMMs. The reason this is done in R is because the statsmodels package in Python does not provide factor level p-values for (generalized) linear mixed effects models. This is tricky because you need to have R and a few packages (lme4, lmerTest, car, afex, and emmeans) installed. This is worth looking into further, as there might be a parameter I have overlooked, or else there could be a different package that fits our needs.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>pain_cutoff</strong></td>
    <td><strong>int</strong></td>
    <td><strong>2</strong></td>
  </tr>
  <tr>
    <td colspan="3">A threshold used as exclusion criteria for participant pain group IDs. Threshold considers the composite pain score (average of intensity, unpleasantness, and interference) This score ranges from 0 to 10, where 0 is no pain and 10 is the worst pain imaginable. The cutoff is used wherein any participant in the no pain group that exceeds this threshold will be excluded from the analysis. And any participant in the pain groups (acute, chronic) that falls below this threshold will be excluded from the analysis. If set to None, no participants will be excluded based on pain.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>depression_cutoff</strong></td>
    <td><strong>int</strong></td>
    <td><strong>10</strong></td>
  </tr>
  <tr>
    <td colspan="3">A threshold used to classify participants into depression groups. This threshold considers the PHQ-8 score, which ranges from 0 to 24. The cutoff is used to classify participants into the depression group if their score is equal to or above this threshold. The data (which is determined using the avoid_learning_data_extraction repo) will already have participants classified into the depression group, but this parameter overrides that classification (because the data incorrectly used a cutoff of > 10).</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>rolling_mean</strong></td>
    <td><strong>int</strong></td>
    <td><strong>5</strong></td>
  </tr>
  <tr>
    <td colspan="3">This parameter determines how many trials to average across the learning phase for plotting the learning curves. This functionality is actually no longer used in the current implementation, because learning curves are now plotted using the binned trials data (early, mid-early, mid-late, late). In the plotting class, the plot_learning_curves function has a binned_trial parameter, which can manually be set to False. If it is set to False (which requires changing the code), then the rolling mean will be used.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>load_stats</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">The GLMMs take a long time to run, so the results are saved in the backend. If this parameter is set to True, the results will be loaded from the backend, instead of running the GLMMs every time this has been run. If this is set to True, but the GLMMs have not been run, it will run the GLMMs anyhow. Make sure to re-run these stats anytime changes are made to the data.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>load_posthocs</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">The posthoc tests can take a long time to add to the report, so the results are saved in the backend. Specifically, tables are turned into png images (which is what takes long), and then uploaded to the report. If this parameter is set to True, the results will be loaded from the backend, and will skip saving to png. Otherwise, it will take the time to save posthocs to png images. If your posthoc stats change, you will need to set this to False for the changes to be reflected in the report.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>load_models</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">If set to True, the models will be loaded from the backend. This is useful if you have run the avoid_learning_rl_models package and want to include the model results in the report. If set to False, the models will not be loaded, and the report will not include any model information. Incorporating data from the avoid_learning_rl_models requires a few steps. You need to first run the analysis with this repo, which will generate data that the avoid_learning_rl_models package can use. This data will be saved in AL/data/ as pain_learning_processed.csv and pain_transfer_processed.csv. You then have to manually copy these files to the avoid_learning_rl_models/RL/data/ directory. After that, you can run the avoid_learning_rl_models package, which will generate the models, and a directory called 'RL/modelling'. This directory can be manually copied to the AL directory, as 'AL/modelling'. Then, you can set this parameter to True, and the models will be loaded from the backend. Sorry this is a bit complicated, with manual steps, but I felt keeping these repos as standalones were better than merging them.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>hide_stats</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">If set to True, the statistics will not be hidden in the report. They will exist but their values will be replaced with 'hidden'. I included this functionality because I believe stats should only be observed once, to mitigate p-hacking. This way, you can hide the stats but still see all other information, such as report details and plots. If you want to see the stats, set this to False.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>hide_posthocs</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">In these analyses, we end up with a lot of posthoc tests, which can clutter the report. It literally adds 40+ pages of tables to the report. This takes a long time to generate, and is not always necessary. Furthermore, it makes the report very large, which is not ideal for sharing. If this parameter is set to True, the posthoc tests will not be included in the report, saving time and space.</td>
  </tr>

  <tr style="background-color:#f0f0f0;">
    <td><strong>verbose</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">If set to True, the pipeline will print additional information to the console during execution. This can be useful for debugging or understanding the flow of the pipeline. If set to False, the pipeline will run more silently without printing additional information.</td>
  </tr>
</table>
