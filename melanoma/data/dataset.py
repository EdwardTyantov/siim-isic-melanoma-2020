import os, sys, cv2, torch
from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SIIMDataset(Dataset):
    def __init__(self, data_path, df_path, transform=None, is_test=False):
        super(SIIMDataset).__init__()
        self.__data_path = data_path
        self.is_test = is_test
        self.__dirname = os.path.join(data_path, is_test and 'test' or 'train')
        self.df = pd.read_csv(df_path) if isinstance(df_path, str) else df_path
        self.transform = transform
        self.feature_names = ['sex', 'age_approx', 'anatom_site_general_challenge'] # TODO:
        # anatom_site_general_challenge nan = 0 ... 0
        self.target_features = ['diagnosis', 'benign_malignant'] # TODO: diagnosis mapping unknown = 0...0
    
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
        
        # TODO: str/nans nahuy
        item = {'image': img, 'path': image_name} # 'features': [meta[fname] for fname in self.feature_names]
        if not self.is_test:
            item['target'] = torch.FloatTensor([meta['target']])
            #item['target_features'] = [meta[fname] for fname in self.target_features]
        
        return item
    
    # @staticmethod
    # def collate_fn(batch):
    #     res = defaultdict(list)
    #     for sample in batch:
    #         for k,v in sample.items():
    #             if k
    #
    #
    #
    #         images.append(torch.from_numpy(sample["image"].transpose((2, 0, 1))).float())
    #         seqs.extend(sample["seq"])
    #         seq_lens.append(sample["seq_len"])
    #     images = torch.stack(images)
    #     seqs = torch.Tensor(seqs).int()
    #     seq_lens = torch.Tensor(seq_lens).int()
    #     batch = {"images": images, "seqs": seqs, "seq_lens": seq_lens}
    #     return batch
        
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



if __name__ == '__main__':
    sys.exit(resize())
