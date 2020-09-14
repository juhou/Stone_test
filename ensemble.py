import pandas as pd
import numpy as np
from glob import glob
import argparse
import os

'''
# 모델 앙상블

총 18개 학습한 모델을 predict를 통해 평가한 결과가 저장된 csv파일
      ./subs/*.csv
를 읽어온 뒤, 모든 값의 평균을 내서 결과를 평가함

# When ensembling different folds, or different models,
# we first rank all the probabilities of each model/fold,
# to ensure they are evenly distributed.
# In pandas, it can be done by df['pred'] = df['pred'].rank(extract)

'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='./subs')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # 폴더에서 csv읽어오기
    subs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.sub_dir, '*csv')))]
    sub_probs = [sub.target.rank(pct=True).values for sub in subs]

    # 균등 가중치
    ensem_number = len(sub_probs)
    wts = [1/ensem_number]*ensem_number

    #
    sub_ens = np.sum([wts[i]*sub_probs[i] for i in range(len(wts))], axis=0)

    # 앙상블 최종 결과를 저장함
    df_sub = subs[0]
    df_sub['target'] = sub_ens
    df_sub.to_csv(f"final_sub1.csv",index=False)
