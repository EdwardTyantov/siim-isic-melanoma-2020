import os, sys, cv2, torch, operator, math
import numpy as np, pandas as pd
from torch.utils.data import Dataset


MAP_NAMES = ('sex', 'anatom_site_general_challenge', 'diagnosis')


def init_mapping(train_df, min_num=5):
    mapping_dict = {fname:{} for fname in MAP_NAMES}
    for feature_name in MAP_NAMES:
        if feature_name == 'diagnosis':
            mapping_dict[feature_name]['unknown'] = -1
        cnt = 0
        for key, value in sorted(train_df[feature_name].value_counts().items(), key=operator.itemgetter(0)):
            if not isinstance(key, str):
                continue
            if key in mapping_dict[feature_name]:
                continue
            if min_num > 0 and value < min_num:
                print(f'Patching mapping: feature_name={feature_name}, key={key} with unknown')
                mapping_dict[feature_name][key] = mapping_dict[feature_name]['unknown']
            else:
                mapping_dict[feature_name][key] = cnt
                cnt += 1
        
    return mapping_dict


class SIIMDataset(Dataset):
    def __init__(self, data_path, df_path, mapping_dict, transform=None, is_test=False):
        super(SIIMDataset).__init__()
        self.__data_path = data_path
        self.is_test = is_test
        self.__dirname = os.path.join(data_path, is_test and 'test' or 'train')
        self.df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
        self.transform = transform
        self.feature_names = ['sex', 'age_approx', 'anatom_site_general_challenge']
        self.mapping_dict = mapping_dict
        if not is_test:
            dw = {self.mapping_dict['diagnosis'][k]: 1 / count for k, count in self.df['diagnosis'].value_counts().items()}
            dw.pop(-1)
            self.diagnosis_weights = torch.FloatTensor([v/sum(dw.values()) for k,v in sorted(dw.items())])
        
    def __getitem__(self, idx):
        meta = self.df.iloc[idx]

        image_name = os.path.join(self.__dirname, meta['image_name'] + '_small.jpg')
        if not os.path.exists(image_name):
            # pick original
            image_name = os.path.join(self.__dirname, meta['image_name'] + '.jpg')
        img = cv2.imread(image_name)
        
        if self.transform is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(image=img)['image']
        
        item = {'image': img, 'path': image_name}
        # for fname in self.feature_names:
        #     value = meta[fname]
        #     if isinstance(value, float) and math.isnan(value):
        #         value = -1
        #     elif fname in MAP_NAMES:
        #         value = self.mapping_dict[fname][value]
        #     item[fname] = value

        if not self.is_test:
            item['target'] = torch.FloatTensor([meta['target']])
            item['diagnosis'] = self.mapping_dict['diagnosis'][meta['diagnosis']]
        
        return item
        
    def __len__(self):
        return len(self.df)


def resize():
    sys.path.insert(0, '..')
    from config import TRAIN_CSV, TEST_CSV, IMAGE_DIR
    df = pd.read_csv(TEST_CSV)
    ds = SIIMDataset(IMAGE_DIR, df, is_test=1)
    print(len(ds))
    r = ds[101]
    print(r['image'].shape)
    
    #
    resize_to = 256
    for i in range(len(ds)):
        temp = ds[i]
        img = temp['image']
        path = temp['path']
        new_path = path.find('small') == -1 and path[:-4] + '_small' + '.jpg' or path
        if os.path.exists(new_path):
            print('already done')
            #continue
        h, w = img.shape[:2]
        if h <= resize_to:
            continue
        k = resize_to/h
        h_, w_ = resize_to, int(w*k)
        img_resized = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_AREA)
        print(i, img.shape, '->', (h_, w_), path, '->', new_path)

        cv2.imwrite(new_path, img_resized)


def main():
    sys.path.insert(0, '..')
    from config import TRAIN_CSV, TEST_CSV, IMAGE_DIR
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print('train')
    mapping = init_mapping(train_df)
    print(mapping)
    ds = SIIMDataset(IMAGE_DIR, train_df, mapping, is_test=0)
    print(len(train_df))
    print(ds[0])
    print('test')
    print(len(test_df))
    ds = SIIMDataset(IMAGE_DIR, test_df, mapping, is_test=1)
    print(ds[1])
    

    feature_names = ['sex', 'anatom_site_general_challenge', 'diagnosis', 'age_approx']
    for fname in feature_names:
        cnt = 0
        for k in train_df[fname]:
            if isinstance(k, float) and math.isnan(k):
                cnt += 1

        print(fname, cnt)


if __name__ == '__main__':
    sys.exit(main())
