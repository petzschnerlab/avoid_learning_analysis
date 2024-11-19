# ===SETUP=== #
# GitHub Repo for Data Analysis
https://github.com/petzschnerlab/SOMA_avoidance_learning

# Data Location
BM_Carney_Petzschner_Lab\SOMAStudyTracking\SOMAV1\database_exports\avoid_learn_prolific

# ===METHODS=== #
# Research Plan:
Avoidance Learning in Chronic Pain Patients (Behavioural).

This document is meant to serve as a method of outlining the research plan for the project Avoidance Learning in Acute and Chronic Pain Patients (Behavioural). It will outline the research goals of the project and the steps taken to achieve those goals.

This project will focus on behavioural findings, which includes descriptive and inferential statistics on accuracy and reaction times as well as the computational modeling of these data using a context-dependent reinforcement learning model and baseline reinforcement learning models.

The overarching goal of the project is to determine whether there are differences in learning mechanisms between chronic pain patients and healthy controls. The project is modeled after Vandendriessche et al., 2023 study that investigated differences in learning between depressed individuals and healthy controls. Specifically, this study found that depressed patients had impaired reward learning in a low rewarding context in contrast to healthy controls as well as in contrast to their learning in a high rewarding context. Computational modeling revealed that this impairment is a heightened sensitivity to feedback within the low rewarding context (i.e., higher learning rates compared to control). This signifies depressed individuals have a heightened sensitivity to negative outcomes. The current project will seek to replicate the analyses conducted by this study but in the context of pain.

This project will also serve as a precursor to a project incorporating EEG with the same experimental paradigm, allowing neural differences in learning to be investigated.

# Research Goals
- Determine whether acute and/or chronic pain patients have impaired learning mechanisms compared to healthy controls across rewarding and punishing contexts
- - i.e., whether learning curves for both contexts differed from healthy controls.
- Determine whether acute and/or chronic pain patients have impaired generalization of learned values across novel contexts
- - i.e., whether choice rates within the transfer phase differed from healthy controls
- Determine whether any impairments can be explained by free parameters of computational reinforcement learning models
- - Model of interest: RELATIVE-RL (context-dependent), different learning rates per context
- - Model benchmark: ABSOLUTE-RL (context-independent), same learning rate across context (traditional Q-Learning model)

# Experimental protocols
The current initiative is to conduct data analysis on previously collected data. Thus, this section is not currently applicable. None-the-less, the experimental protocols will need to be written up for any manuscript and should be added here when done. The following is a repo of the experiment to be described: https://github.com/petzschnerlab/soma_avoid_eeg

# Data analysis
All references refer to Vandendriessche et al., 2023 (Van et al hereafter)

## Behavioural Analyses
Focus is on accuracy rates, not reaction times as the previous literature found no reaction time effects and due to the use of prolific, reaction times recorded in this study may vary widely depending on the tech (e.g., computer, internet strength) used per participant.

### Steps:
- **Learning Phase**:
    1. Thorough investigation of the data
    2.  Replicate Figure 2
    3.  Conduct all statistical analyses
    4.  Learning Phase:GLMM of correct choice rates
        - Determine if accuracy was above chance (intercept)
            - Van et al: Found significant effect
        - Determine effects of context (reward, punish), group (controls, acute pain, chronic pain), and their interaction.
            - Van et al: Only signifiance in the interaction, post-hocs determining this was driven by impaired learning in patients

- **Transfer Phase**: GLMM of choice rates (3 levels, those containing A when D not present, those containing D when A not present, and the rest)
    1. Determine if choice rate was above chance (intercept)
        - Van et al: Found significant effect
    2. Determine effects of context (reward, punish), group (controls, acute pain, chronic pain), and their interaction.
        - Van et al: Significant group and interaction, but not context

## Statistical Analyses Described 
### Demographics
- **Vandendriessche et al. 2023:** 
    - T-Tests:
        - Age matched
        - Education matched
        - Depression scales: LOt-R, usual optimism, current optimism 

### Learning Phase
- **Palminteri et al., 2015:**
    - Software: Matlab Statistical Toolbox
    - ANOVA: Accuracy ~ f_information*f_valence
        - Accuracy: Percentage of correct choices (cont scale)
        - Feedback Information: Partial vs complete information
        - Feedback Valence: Reward vs Punishment contexts
    - Post-Hocs: 
        - Two-sided, one sample t-test of context
            -Context: reward/partial, reward/complete, punishment/partial and punishment/complete
    - **Findings:** 
        - Overall performance above chance (via t-test vs 50%)
        - Significant feedback information (not feedback valence/context or interaction)
            - Post-Hocs: Complete feedback > partial feedback in both contexts 
    
- **Geana, et al, and Frank, 2021:**
    - Software: SPSS
    - ANOVA: Accuracy ~ Timee x group x condition
        - Assumptions: They used Hyunh-Feldt correction for violating assumption of sphericity
        - Accuracy: moving time window of 20 trials **look into this**
        - Time: Trial number? **look into this**
        - Group: Bipolar disorder, schizophrenia with meds, schizophrenia without meds, controls
        - Condition: Context, reward or punish
    - **Findings:** 
        - Significant group effect, controls had higher accuracy than all groups. No interaction btw time and group

- **Gold et al., 2012:**
    - ANOVA 1: Accuracy ~ group x f_valence x probability x learning block
        - Correction: Hyun-Feldt correction was applied if assumption of sphericity was violated
        - Group: controls, Schizophrenia-LNS (low negative symptom), Schizophrenia-HNS (high neg sympt)
        - Feedback Valence: Context -> reward punishment
        - Probability: 90% change of winning vs 80%
        - Learning Block: 4 learning blocks, time of learning
    - ANOVA 2: Accuracy (B4) ~ valence x probability x group
        - Accuracy: Block 4 only
    - **Findings:**
        - ANOVA 1
            - Significant probability (90% > 80%)
            - Significant learning block (inc acc over time)
            - Significant probability x group interaction
                - HNS does not show greater performance for probability but other groups do
            - Significant valence x learning block interaction
        - ANOVA 2
            - Significant probability 
            - Significant group x valence interaction
                - HC greater learning on 90% than HNS group
                - Better learning from gain than punish in HC vs HNS group

- **Vandendriessche et al. 2023:** 
    - Software: R glmer
    - GLMM: Accuracy ~ group*context + (1|participant), link: binomial
        - Accuracy: 0, 1 (**this is odd, must be an average?**)
        - Group: control, patients
        - Context: rich, poor
    - Post-Hocs: 
        - Comparing marginal means to zero
        - Tukey correction
    - **Findings:** 
        - Overall performance above chance (intercept)
        - Significant interaction of group*context
            - Post-Hocs: Effect of context in patients, but not controls (slopes presented)


### Transfer Phase
- **Palminteri et al., 2015:**
    - Software: Matlab Statistical Toolbox
    - ANOVA: Accuracy ~ f_information x f_valence x option_correctness
        - Feedback Information: Partial vs complete information
        - Feedback Valence: Reward vs Punishment contexts
        - Option correctness: **what is this? It is never explained**
    - **Findings:**
        - Significant outcome valence and option_correctness, no effect of feedback information

- **Geana, et al, and Frank, 2021:**
    - Software: SPSS
    - ANOVA: Accuracy ~ group x condition
        - Multiple ANOVAs using 75R-25P vs 75R-75P and 75R-25P vs 75R-25R
    - **Findings:** 
        - Significant condition in both ANOVAs
            - Higher acc in 75R-75P and 75R-25R versus 75R-25P **double check this**
        - 75R-25P condition main effect of group (controls higher accuracy than SZOFF) **not sure how this was conducted**
        - Learned (old) pairings showed main effect of group where controls > all patients.

- **Gold et al., 2012:**
    - ANOVA: Accuracy ~ group
        - Accuracy: percent correct
        - Group: control, LNS, HNS
    - Post-Hocs: 
    - **Findings:** 
        - Difference of accuracy btw groups
            - HC > HNS
            - HC had preference for 75R over 25P but HNS group showed no preference for 75R over 25P?
        - All groups prefered 25P over 25R

- **Vandendriessche et al. 2023:**
    - Software: R glmer
    - GLMM: Accuracy ~ group*condition + (1|participant), link: binomial
        - Accuracy: 0, 1 (**this is odd, must be an average?**)
        - Group: control, patients
        - Condition: best, other (intermediate), worst
            - Best: incudes 75R (but not 75P)
                - 75R vs 25R/25P
            - Other: Does not include 75R or 75P 
                - 25R vs 25P (**maybe also 75R vs 75P?**)
            - Worst: includes 75P (but not 75R)
                - 75P vs 25R/25P
    - Post-Hocs: 
        - Comparing marginal means to zero
        - Tukey correction
    - **Findings:**
        - Overall performance above chance (intercept)
        - Significant group and interaction effects
            - Post-Hocs: Patients were better at seeking 75R than avoiding 75P (controls these were equal)

### Transfer Phase Choice Rate Statistics (*Post-Hocs*)

- **Palminteri et al., 2015**
    - **Choice Rate:*** Calculated as number of times chosen vs all other conditions (4 levels)
    - All between option choices (difference measures) were analyzed using one-sample ttests (no corrections for post-hocs!)
    - Intermediate values (low reward | low punishment) were also compared to all other conditions (high reward, low punish/low reward, high punish)
    - **Results:**
        - low reward not equal to low punish
        - high reward > low punish

- **Vandendriessche et al., 2023**
    - **Choice Rate:** Calculated as number of times chosen vs all other conditions (4 levels)
        - *Note: For the GLMM, calculated as number of times chosen dependent on whether it was in contrast to High Reward, Moderate Values, or High Punish (3 levels)*
        - Compared marginal means to zero with Tukey post-hoc corrections
    - **Results:** 
        - *Note, this task had rich and poor conditions not reward/punish but it will be written as reward/punish here for consistency*
        - Controls: High reward (choice) = High punish (avoid)
        - Patients: High reward (choice) > high punish (avoid) -> better at seeking reward than avoiding loss
        - All groups: High reward vs low punish

- **Gold et al., 2012**
    - Group differences~High Reward - Low Punish (testing the relative value)
        - Healthy: Preference for High Reward, Clincial (Schz): equal preference?
    - Group differences~Low Reward - Low punish (testing the absolute value)
        - All groups: Low punish > low reward


- **Specific Analyses**
    - High Reward vs Low Punish (Palminteri et al., 2015, Gold et al., 2012, Vandendriessche et al., 2023)
    - Low Reward vs Low Punish (Palminteri et al., 2015, Gold et al., 2012)

## Computational Modelling
### Models:
- Develop the Q-learning benchmark RL model (Palminteri et al., 2015)
- Develop the relative RL model (Palminteri et al., 2015)
- Develop the relative +- RL model (Lefebvre et al., 2017)
- Hybrid actor-critic-q-learning model (Gold et al., 2012)

### Analyses
- Run model simulations to validate models (fig 3A)
- Fit models to empirical findings (fig 3B)
- Extract learning rates and run statistics
- Extract temperature and run statistics

# Literature Reviews
- *[Palminteri et al. - 2015](https://www.nature.com/articles/ncomms9096.pdf)
- *[Vandendriessche et al. - 2023](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4A99B6789148211973379AB7A8A81036/S0033291722001593a.pdf/div-class-title-contextual-influence-of-reinforcement-learning-performance-of-depression-evidence-for-a-negativity-bias-div.pdf)
- Bavard et al. - 2018
- [Gold - 2012](https://jamanetwork.com/journals/jamapsychiatry/fullarticle/1107396)
- [Geana et al. - 2021](https://www.sciencedirect.com/science/article/pii/S2451902221001154?casa_token=O8ktVYVJGI8AAAAA:omzWr6OSLP0ujwii_z51yjP_hdO0EUtzqrkTbj8Fh7hkbSaYVqq5FR9-ecwyjUvKuBUqdlrwfYo) 
- https://youtu.be/a483TKDJ7ss?si=LrVM__enUvdC0KSY
- https://www.youtube.com/watch?v=etQVRd6N8dM&t=697s
- [Lefebvre et al. - 2017](https://www.unicog.org/publications/LefebvreLebretonMeynielBourgeois-GirondePalminteri_2017_NHB_Behavioral-and-neural-characterization-of-optimistic-reinforcement-learning.pdf)

# ===RESULTS=== #

## General Conclusions

- **Learning Phase**
    - *Accuracy:* Reward context has greater accuracy than punish context 
    - *Reaction Time:* 
        - Chronic pain patients are slower than other groups
        - Reward context is slower than Punish context
    - **Summary:** Reward context had higher accuracy and was slower for selection than the punish context. The chronic pain patients took longer to respond in this task overall than the acute and no pain groups.
        - **Conclusion:** People spend more time to be more accurate in rewarding contexts. Chronic pain patients take more time than other groups to reach the same accuracy.

- **Transfer Learning**
    - *Accuracy:* High reward was chosen over low reward across all groups
    - *Accuracy:* Low reward was preferred over low punish only for the no pain group; the acute and chronic pain groups showed equal preference
    - *Reaction Time:*
        - Chronic pain patients slower than both other groups
        - High reward was faster than low punish but no difference between low reward and low punish
        - There's no difference in RTs for pain groups between low reward and low punish, but there is a difference for the no pain group.
    - **Summary:** The no pain group preferred (and were quicker to select) the low reward over the low punish conditions. In contrast, the pain groups (chronic, acute) had no preference (or reaction time differences) for the low reward versus the low punish conditions.
        - **Conclusion**: The low reward vs low punishment condition is meant to test whether a person judges their stimuli based on objective values (i.e., the amount of money earned where low reward > low punish), contextual values (i.e., which was more rewarding in their respective contexts where low punish > low reward). As such, the no pain group judged stimuli using their objective values (low reward > low punish) but the pain groups used a mixture of the two dimensions (i.e., combining low reward > low punish + low punish > low reward results in low reward = low punish).

## Learning Phase

### Accuracy
- **Group:** No effect
- **Context effect:** Learning was faster in the reward vs punish condition
- **Trial:** Learning increased across trials
- **GroupxTrial:** Effect, but post-hocs show nothing interesting
    - Chronic vs No Pain: No effects
        - Early: No effect
        - Mid-Early: No effect
        - Mid-Late: No effect
        - Late: No effect
    - Chronic vs Acute: No effect
        - Early: No effect
        - Mid-Early: No effect
        - Mid-Late: No effect
        - Late: No effect
    - No Pain vs Acute: No effect
        - Early: No effect
        - Mid-Early: No effect
        - Mid-Late: No effect
        - Late: No effect

### Reaction Time
- **Group:** Effect
    - Chronic vs No Pain: Chronic Slower than No Pain
    - Chronic vs Acute: Chronic slower than acute
    - No pain vs Acute: No difference
- **Context:** Effect
    - Reward Slower than Punish Context

## Transfer Phase

### Choice Rate
- **Group:** No effect
- **Condition:**: Effect
    - High reward > low punish
    - Low reward > low punish
- **GroupxCondition:**: Effect
    - Low Reward-Low Punish:
        - No Pain: Difference
        - Acute: No difference
        - Chronic: No difference
        
        - No Pain vs Acute: Greater difference for no pain
        - Chronic vs No Pain: Greater difference for no pain
        - Chronic vs Acute: No difference
        
        - **Conclusion**: Pain groups showed equal value to both low reward, low punish conditions, but no pain group showed preference to the reward condition.

    - High Reward-Low Punish
        - No Pain: Difference
        - Acute Pain: Difference
        - Chronic Pain: Difference

        - No Pain vs Acute: Greater difference in no pain 
        - Chronic vs No Pain: No difference
        - Chronic vs Acute: No difference

        - **Conclusion:** Although a difference in no pain vs acute pain, there is no real interesting patterns here.

### Reaction Time
- **Group:** Effect
    - Chronic vs No Pain: Chronic slower than no pain
    - Chronic vs Acute: Chronic slower than acute
    - No pain vs Acute: No difference
- **Context:** Effect
    - High Reward vs Low punish: High reward faster than low punish
    - Low Reward vs low punish: Equal
- **GroupxContext:** Effect
    - High Reward - Low Punish
        - Chronic vs No Pain: No difference
        - Chronic vs Acute: No difference
        - No Pain vs Acute: Bigger difference in no pain than acute
    - Low Reward - Low Punish
        - Chronic vs No Pain: Difference in no pain, but no diff in chronic
        - Chronic vs Acute: No difference
        - No Pain vs Acute: Difference in no pain, but no diff in acute

# ===NOTES=== #
# Project notes

## Lack of Replication in Transfer Learning Effect

Our behavioural findings do not replicate the past literature (e.g., Palminteri's work). Specifically, in the transfer phase they find that people tend to select the 25% Punish symbols more often than the 25% Reward symbols. We find the opposite patterns in that healthy controls select the 25% reward symbols more often than the 25% Punish symbols and acute/chronic pain patients tend to select them both equally (see Figure 4 on the report). To try and decipher why this might be, I began with investigating task design. The tasks are quite similar but do have some differences (e.g., the novel symbol in the transfer phase) that could potentially cause the lack of replication, but I have found something in the task instructions that seems like a possible candidate. 

Palminteri's work uses these instructions for the transfer phase:
Now you will start the second session. The aim of this session is to find out which was the most advantageous symbol in the previous session. The symbols presented together in a given trial have not necessarily been presented together in the previous session. Your task is to figure out which one won more points or lost less points. If you have no idea, try to have a guess. Your score for this sessions will correspond to the sum of the values of chosen symbols. You will know your score only at the end of the session, and not trial-by-trial

Our task uses these instructions for the transfer phase:
In this next part, you will see the same knights as before, but they will be shown in new pair combinations. Again, your job will be to select the knight you would like to join your team. As you make your choices, you will not receive any feedback after your choice. You should still choose the knight you think is better on each trial. Your choices will still contribute to your performance bonus.

Note the bolded sections of each quote. To me, it seems like Palminteri's instructions elicit a context-dependent perspective by equating both best options in the reward and punish contexts (i.e., they say to choose the most rewarding and the least punishing symbols whenever possible). In our task, it seems like the instructions rather elicit a context-independent perspective by indicating to choose the BEST knight as possible (because that's the one you would want on your team). 