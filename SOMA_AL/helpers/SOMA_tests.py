import os 
import numpy as np
import matplotlib.pyplot as plt

class SOMATests:

    """
    Class to run tests for the SOMA project
    """

    def run_tests(self):
        self.test_trial_counts()
        if self.tests == 'extensive':
            self.test_plot_learning_accuracy(self.test_rolling_mean)

    def test_trial_counts(self):

        #for each articiant plot the trial indexes as the y axis using matlpotlib
        for participant in self.learning_data['participant_id'].unique():
            participant_data = self.learning_data[self.learning_data['participant_id'] == participant]
            participant_trial_counts = []
            for context in participant_data['context_val_name'].unique():
                context_data = participant_data[participant_data['context_val_name'] == context]
                participant_trial_counts.append(context_data.shape[0])

            if participant_trial_counts[0] != 48 or participant_trial_counts[1] != 48:
                raise ValueError(f'Participant {participant} has incorrect number of trials: {participant_trial_counts}') 
            
    def test_plot_learning_accuracy(self, rolling_mean=None):

        #Create folder for plots if non existent
        if not os.path.exists('SOMA_AL/plots/tests'):
            os.makedirs('SOMA_AL/plots/tests')
        if not os.path.exists('SOMA_AL/plots/tests/no pain'):
            os.makedirs('SOMA_AL/plots/tests/no pain')
        if not os.path.exists('SOMA_AL/plots/tests/acute pain'):
            os.makedirs('SOMA_AL/plots/tests/acute pain')
        if not os.path.exists('SOMA_AL/plots/tests/chronic pain'):
            os.makedirs('SOMA_AL/plots/tests/chronic pain')
        if not os.path.exists('SOMA_AL/plots/tests/healthy'):
            os.makedirs('SOMA_AL/plots/tests/healthy')
        if not os.path.exists('SOMA_AL/plots/tests/depressed'):
            os.makedirs('SOMA_AL/plots/tests/depressed')

        for i, group in enumerate(self.group_labels):
            folder_name = f'SOMA_AL/plots/tests/{group}/'
            group_data = self.learning_data[self.learning_data[self.group_code] == group]
            for participant in group_data['participant_id'].unique():
                participant_data = group_data[group_data['participant_id'] == participant]
                for context_index, context in enumerate(['Reward', 'Loss Avoid']):
                    context_data = participant_data[participant_data['context_val_name'] == context]['accuracy']
                    if rolling_mean is not None:
                        context_data = context_data.rolling(rolling_mean).mean()
                    plt.plot(np.arange(1,context_data.shape[0]+1) ,context_data, color=['#B2DF8A', '#FB9A99'][context_index], label=['Reward' if context == 'Reward' else 'Punish'])

                plt.ylim(-5, 105)
                plt.title(f'{participant.capitalize()}')
                plt.xlabel('Trial Number')
                plt.ylabel('Accuracy')
                plt.legend(loc='lower right', frameon=False)

                #Save the plot
                plt.savefig(f'{folder_name}{participant}.png')

                #Close figure
                plt.close()


            
