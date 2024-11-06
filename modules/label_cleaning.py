import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from argument import DataCentric
class label_cleaner:
    def __init__(self):
        self.args = DataCentric()
        self.origin = pd.read_csv(self.args.train_route)
    
    def find_issues(self, df):
        labels = self.origin['target']
        logits = np.vstack(df['logits'])
        self.label_issues = find_label_issues(labels = labels,
                                         pred_probs = logits,
                                         return_indices_ranked_by = 'self_confidence')
        print('발견한 라벨이슈의 개수 :', len(self.label_issues))
        self.origin.loc[self.label_issues, 'target'] = df.loc[self.label_issues, 'target']

    def to_csv(self):
        self.origin[['ID', 'text', 'target']].to_csv('processed_train.csv', index = False)
    