import pandas as pd
import numpy  as np
import random
random.seed(42)
from   scipy.special import expit, logit
 
almost_zero = 1e-04
almost_one  = 1-almost_zero

df1 = pd.read_csv("./ensembling_submission_01.csv").rename(columns={'deal_probability': 'dp1'})
df2 = pd.read_csv("./ensembling_submission_02.csv").rename(columns={'deal_probability': 'dp2'})
df3 = pd.read_csv("./ensembling_submission_03.csv").rename(columns={'deal_probability': 'dp3'})
df4 = pd.read_csv("./ensembling_submission_04.csv").rename(columns={'deal_probability': 'dp4'})
df5 = pd.read_csv("./ensembling_submission_05.csv").rename(columns={'deal_probability': 'dp5'})
df6 = pd.read_csv("./ensembling_submission_06.csv").rename(columns={'deal_probability': 'dp6'})
df7 = pd.read_csv("./ensembling_submission_07.csv").rename(columns={'deal_probability': 'dp7'})

df = pd.merge(df1, df2, on='item_id')
df = pd.merge(df, df3, on='item_id')
df = pd.merge(df, df4, on='item_id')
df = pd.merge(df, df5, on='item_id')
df = pd.merge(df, df6, on='item_id')
df = pd.merge(df, df7, on='item_id')

scores = [0, 0.2203, 0.2203, 0.2203, 0.2203, 0.2203, 0.2203, 0.2203] # public leaderboard scores


weights = [0, 0, 0, 0, 0, 0, 0, 0]

weights[1] = scores[1] ** 4
weights[2] = scores[2] ** 4
weights[3] = scores[3] ** 4
weights[4] = scores[4] ** 4
weights[5] = scores[5] ** 4
weights[6] = scores[6] ** 4
weights[7] = scores[7] ** 4

print(weights[:])

number1 = df['dp1'].clip(almost_zero,almost_one).apply(logit)  * weights[1]
number2 = df['dp2'].clip(almost_zero,almost_one).apply(logit)  * weights[2]
number3 = df['dp3'].clip(almost_zero,almost_one).apply(logit)  * weights[3]
number4 = df['dp4'].clip(almost_zero,almost_one).apply(logit)  * weights[4]
number5 = df['dp5'].clip(almost_zero,almost_one).apply(logit)  * weights[5]
number6 = df['dp6'].clip(almost_zero,almost_one).apply(logit)  * weights[6]
number7 = df['dp7'].clip(almost_zero,almost_one).apply(logit)  * weights[7]

totalweight = sum(weights)

df['deal_probability'] = ( number1 + number2 + number3 + number4 + number5 + number6 + number7) / ( totalweight )

df['deal_probability']  = df['deal_probability'].apply(expit) 

# Any results you write to the current directory are saved as output.
df[['item_id', 'deal_probability']].to_csv("ensemb_ensemb_01.csv", index=False)