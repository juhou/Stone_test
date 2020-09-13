import os
import re
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class StoneDataset(Dataset):
    '''
        class 내가만든_데이터셋(Dataset):
            def __init__(self, csv, mode, meta_features, transform=None):
                # 데이터셋 초기화

            def __len__(self):
                # 데이터셋 크기 리턴
                return self.csv.shape[0]

            def __getitem__(self, index):
                # 인덱스에 해당하는 이미지 리턴


        self.mode = mode
        # train / valid
        self.use_meta = meta_features is not None
        # 안쓸경우 None
        self.meta_features = meta_features
        # 안쓸경우 None
        self.transform = transform
        # 이미지 트랜스폼
    '''

    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode # train / valid
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        # 메타 데이터를 쓰는 경우엔 image와 함께 텐서 생성
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            # Test 의 경우 정답을 모르기에 데이터만 리턴
            return data
        else:
            # training 의 경우 CSV의 스톤여부를 타겟으로 보내줌
            return data, torch.tensor(self.csv.iloc[index].target).long()


def get_df(kernel_type, data_dir, data_folder, out_dim = 1, use_meta = False, use_ext = False):
    '''
    get DataFrame
    데이터베이스 관리하는 CSV 파일을 읽어오고, 교차 validation을 위해 분할함

    :param kernel_type: argument에서 받은 네트워크 설명 문구
    :param out_dim: 네트워크 출력 개수
    :param data_dir: 데이터 베이스 폴더
    :param data_folder: 데이터 폴더
    :param use_meta: meta 데이터 사용 여부
    :param use_ext: 외부 추가 데이터 사용 여부

    :return:
    :mel_idx 양성을 판단하는 인덱스 번호

    참고사항: tfrecord는 텐서플로우 제공하는 데이터 포멧
    https://bcho.tistory.com/1190
    '''

    # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
    # https://rfriend.tistory.com/250
    df_train = pd.read_csv(os.path.join(data_dir, data_folder, 'train.csv'))

    # 비어있는 데이터 버리고 데이터 인덱스를 재지정함
    # https://kongdols-room.tistory.com/123
    # df_train = df_train[df_train['foobar_index_name'] != -1].reset_index(drop=True)

    # 이미지 이름을 경로로 변환하기 (pd.DataFrame / apply, map, applymap에 대해 공부)
    # http://www.leejungmin.org/post/2018/04/21/pandas_apply_and_map/
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}train', x))  # f'{x}.jpg'

    # 교차 검증을 위해 이미지 리스트들에 분리된 번호를 매김
    patients = len(df_train['patient_id'].unique())
    print(f'사람 인원수 : {patients}')

    # 데이터 인덱스 : fold 번호
    # x번 데이터가 fold 번 분할로 들어간다라는 의미
    # train.py input arg에서 fold를 수정해줘야함
    if 'fold' in kernel_type:
        # k-fold cross-validation
        regex = re.compile(r'\d+fold')
        k = int(regex.search(kernel_type).group().split('fold')[0])
        print(f'Dataset: {k}-fold cross evaluation')

        # 환자id : 분할 번호
        patients2fold = {i: i % k for i in range(patients)}
    else:
        pass
        # 다른 폴딩 샘플
        # person2fold = {
        #     8: 0, 5: 0, 11: 0,
        #     7: 1, 0: 1, 6: 1,
        #     10: 2, 12: 2, 13: 2,
        #     9: 3, 1: 3, 3: 3,
        #     14: 4, 2: 4, 4: 4,
        # }

    df_train['fold'] = df_train['patient_id'].map(patients2fold)

    # 외부 데이터를 사용할 경우 이곳을 구현
    if use_ext:
        pass
        '''
        # 외부 추가 데이터 (external data)
        # data 읽어오기 (pd.read_csv / DataFrame으로 저장)
        ext_data_folder = 'additional_data/'
        df_train2 = pd.read_csv(os.path.join(data_dir, ext_data_folder, 'train.csv'))
        
        # 위의 원본 데이터 입력 예제와 비슷하게 구현하면 됨. 
    
    
        # 특이사항: 출력의 크기에 따라서 레이블을 바꿔야 하는 경우 아래코드 사용
        if out_dim == 2:
            # output 수에 따른 외부 데이터 정리 예시
            df_train['target'] = df_train['target'].apply(lambda x: x.replace('0', '1'))
        elif out_dim == 1:
            pass
        else:
            raise NotImplementedError()
        
        # concat train data
        df_train2['is_ext'] = 1
        df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)
        '''
    else:
        df_train['is_ext'] = 0  # 원본데이터=0, 외부데이터=1

    
    # test data (주의: 현재 테스트가 없어서, 동일한 데이터를 사용함)
    df_test = pd.read_csv(os.path.join(data_dir, data_folder, 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{data_folder}test', x)) # f'{x}.jpg'

    # 메타 데이터를 사용하는 경우 (나이, 성별 등)
    # 아래쪽 함수 참조
    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    # class mapping - 정답 레이블을 기록
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.target.unique()))}
    df_train['target'] = df_train['target'].map(diagnosis2idx)

    # CSV 기준 정상은 0, 비정상은 1
    # 여기서는 stone 데이터를 타겟으로 삼음
    target_idx = diagnosis2idx[1]

    return df_train, df_test, meta_features, n_meta_features, target_idx



def get_meta_data(df_train, df_test):
    '''
    메타 데이터를 사용할 경우
    (이미지를 표현하는 정보: 성별, 나이, 사람당 사진수, 이미지 크기 등)

    :param df_train:
    :param df_test:
    :return:
    '''

    # One-hot encoding of targeted feature
    concat = pd.concat([df_train['targeted_feature'], df_test['targeted_feature']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)

    # Sex features - 1과 0으로 변환
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)

    # Age features - [0,1] 사이의 값으로 변환
    df_train['age'] /= 90
    df_test['age'] /= 90
    df_train['age'] = df_train['age'].fillna(0)
    df_test['age'] = df_test['age'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)

    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)

    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    # df_train.columns에서
    meta_features = ['sex', 'age', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features



def get_transforms(image_size):
    '''
    albumentations 라이브러리 사용함
    https://github.com/albumentations-team/albumentations

    TODO: FAST AUTO AUGMENT
    https://github.com/kakaobrain/fast-autoaugment
    DATASET의 AUGMENT POLICY를 탐색해주는 알고리즘

    TODO: Unsupervised Data Augmentation for Consistency Training
    https://github.com/google-research/uda

    TODO: Cutmix vs Mixup vs Gridmask vs Cutout
    https://www.kaggle.com/saife245/cutmix-vs-mixup-vs-gridmask-vs-cutout

    '''
    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),

        # one of 의 경우 하나를 랜덤하게 뽑아서 쓰게 해줌
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val
