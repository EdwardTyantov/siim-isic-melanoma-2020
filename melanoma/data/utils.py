import logging
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from .dataset import SIIMDataset, init_mapping
from .transform import factory


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class Datasets(object):
    def __init__(self, image_dir, train_csv, test_csv, transform_name, image_size, p=0.5, val_split=0.2):
        self.transform_train, self.transform_val, self.transform_test = factory(transform_name, image_size, p)
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        mapping = init_mapping(train_df)

        patient_means = train_df.groupby(['patient_id'])['target'].mean()
        patient_ids = train_df['patient_id'].unique()
        # TODO: add K-fold
        train_idx, val_idx = train_test_split(np.arange(len(patient_ids)), stratify=(patient_means > 0),
                                              test_size=val_split)
        pid_train, pid_val = patient_ids[train_idx], patient_ids[val_idx]
        tr_df = train_df[train_df['patient_id'].isin(pid_train)]
        val_df = train_df[train_df['patient_id'].isin(pid_val)]
        
        self.train_dataset = SIIMDataset(image_dir, tr_df, mapping, self.transform_train, is_test=False)
        self.val_dataset = SIIMDataset(image_dir, val_df, mapping, self.transform_val, is_test=False)
        self.test_dataset = SIIMDataset(image_dir, test_df, mapping, self.transform_test, is_test=True)
    
        logger.info('Length of datasets train/val/test=%d/%d/%d', len(self.train_dataset), len(self.val_dataset),
                    len(self.test_dataset))
