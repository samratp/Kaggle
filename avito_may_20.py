import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import random
from tqdm import tqdm

random.seed(2018)
print("Data:\n",os.listdir("./Avito"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

print("\nData Load Stage")
training = pd.read_csv('./Avito/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
traindex = training.index
testing = pd.read_csv('./Avito/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
#del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print(df.head())
print(df.info())

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-10,inplace=True)
df["image_top_1"].fillna(-10,inplace=True)

print("\nCreate Time Variables")
df["activation_weekday"] = df['activation_date'].dt.weekday
df["activation_weekday_of_year"] = df['activation_date'].dt.week
df["Day_of_Month"] = df['activation_date'].dt.day

predictors=[]

## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #print('predictors',predictors)
    gc.collect()
    return( df )
    
##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #print('predictors',predictors)
    gc.collect()
    return( df )
    
### Below a function is written to extract cumulative count feature  from different cols    
def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #print('predictors',predictors)
    gc.collect()
    return( df )
    
### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #print('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
    #print('predictors',predictors)
    gc.collect()
    return( df )

#--------------------------------#
# 1. Simple aggregated features  #
#--------------------------------#
# https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# map() function returns a list of the results after applying the given function to each item of a given iterable 
# Get mean and standard deviation of price by different groups
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type', 'Day_of_Month', 'item_seq_number', 'activation_weekday']

#Get mean price by different groups
for c in tqdm(agg_cols):
    gp = df.groupby(c)['price']
    mean = gp.mean()
    std = gp.std()
    var = gp.var()
    median = gp.median()
    cumcount = gp.cumcount()
    df[c + '_price_avg'] = df[c].map(mean)
    df[c + '_price_std'] = df[c].map(std)
    df[c + '_price_var'] = df[c].map(var)
    df[c + '_price_median'] = df[c].map(median)
    df[c + '_price_cumcount'] = df[c].map(cumcount)
    
gc.collect()
print ("Step 1 finished")


#--------------------------------#
# 2. Simple aggregated features  #
#--------------------------------#
# https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# map() function returns a list of the results after applying the given function to each item of a given iterable 
# Get mean and standard deviation of price by different groups
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type','Day_of_Month', 'item_seq_number', 'activation_weekday']

#Get mean price by different groups
for c in tqdm(agg_cols):
    gp = df.groupby(c)['image_top_1']
    mean = gp.mean()
    std = gp.std()
    var = gp.var()
    median = gp.median()
    cumcount = gp.cumcount()
    df[c + '_image_top_1_avg'] = df[c].map(mean)
    df[c + '_image_top_1_std'] = df[c].map(std)
    df[c + '_image_top_1_var'] = df[c].map(var)
    df[c + '_image_top_1_median'] = df[c].map(median)
    df[c + '_image_top_1_cumcount'] = df[c].map(cumcount)
    
gc.collect()
print ("Step 2 finished")


#--------------------------------#
# 3. Simple aggregated features - Region  #
#--------------------------------#
# https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# map() function returns a list of the results after applying the given function to each item of a given iterable 
# Get mean and standard deviation of price by different groups
agg_cols = [['region', 'city'], ['region', 'parent_category_name'], ['region', 'category_name'], ['region', 'image_top_1'],
            ['region', 'user_type'], ['region', 'Day_of_Month'], ['region', 'item_seq_number'], ['region', 'activation_weekday']]

#Get mean price by different groups
for c in tqdm(agg_cols):
    print(c)
    gp = df.groupby(c)['price']
    mean = gp.mean()
    std = gp.std()
    var = gp.var()
    median = gp.median()
    cumcount = gp.cumcount()
    df[c + '_price_avg'] = df[c].map(mean)
    df[c + '_price_std'] = df[c].map(std)
    df[c + '_price_var'] = df[c].map(var)
    df[c + '_price_median'] = df[c].map(median)
    df[c + '_price_cumcount'] = df[c].map(cumcount)
    
gc.collect()
print ("Step 3 finished")

'''
#--------------------------------#
# 2. Target Encoding             #
#--------------------------------#
# https://www.kaggle.com/classtag/lightgbm-with-mean-encode-feature-0-233
# map() function returns a list of the results after applying the given function to each item of a given iterable 
# Get mean and standard deviation of price by different groups
agg_cols = ['region', 'city', 'parent_category_name', 'category_name', 'image_top_1', 'user_type', 'item_seq_number', 'activation_weekday']

#Get mean price by different groups
for c in tqdm(agg_cols):
    gp = training.groupby(c)['deal_probability']
    mean = gp.mean()
    std  = gp.std()
    var = gp.var()
    median = gp.median()    
    df[c + '_deal_probability_avg'] = df[c].map(mean)
    df[c + '_deal_probability_std'] = df[c].map(std)
    df[c + '_deal_probability_var'] = df[c].map(var)
    df[c + '_deal_probability_median'] = df[c].map(median)    

training.drop("deal_probability",axis=1, inplace=True)
'''

# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date<=pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date>=pd.to_datetime('2017-04-08')].index
df.drop(["activation_date","image"],axis=1,inplace=True)

print("\nEncode Variables")
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1"]
print("Encoding :",categorical)

print(df.head())

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 
df['text_feat'] = df.apply(lambda row: ' '.join([
    str(row['param_1']), 
    str(row['param_2']), 
    str(row['param_3'])]),axis=1) # Group Param Features
df.drop(["param_1","param_2","param_3"],axis=1,inplace=True)

# Meta Text Features
textfeats = ["description", "text_feat", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('nicapotato') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_chars'] = df[cols].apply(len) # Count number of Characters
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words

gc.collect();
print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}
def get_col(col_name): return lambda x: x[col_name]
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=16000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('text_feat',CountVectorizer(
            ngram_range=(1, 2),
            #max_features=7000,
            preprocessor=get_col('text_feat'))),
        ('title',TfidfVectorizer(
            ngram_range=(1, 2),
            **tfidf_para,
            #max_features=7000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()
vectorizer.fit(df.loc[traindex,:].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))
gc.collect();

# Drop Text Cols
df.drop(textfeats, axis=1,inplace=True)

# Dense Features Correlation Matrix
f, ax = plt.subplots(figsize=[10,7])
sns.heatmap(pd.concat([df.loc[traindex,[x for x in df.columns if x not in categorical]], y], axis=1).corr(),
            annot=False, fmt=".2f",cbar_kws={'label': 'Correlation Coefficient'},cmap="plasma",ax=ax, linewidths=.5)
ax.set_title("Dense Features Correlation Matrix")
plt.yticks(rotation=90)
plt.savefig('./May_20/correlation_matrix_v4.png')

print("Modeling Stage")
# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")

# Training and Validation Set
"""
Using Randomized train/valid split doesn't seem to generalize LB score, so I will try time cutoff
"""
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'max_depth': 12,
    'num_leaves': 37,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    'learning_rate': 0.019,
    'verbose': 0
}  

# LGBM Dataset Formatting 
lgtrain = lgb.Dataset(X_train, y_train,
                feature_name=tfvocab,
                categorical_feature = categorical)
lgvalid = lgb.Dataset(X_valid, y_valid,
                feature_name=tfvocab,
                categorical_feature = categorical)

# Go Go Go
modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=16000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=200,
    verbose_eval=200
)

# Feature Importance Plot
f, ax = plt.subplots(figsize=[20,30])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('./May_20/feature_import_v4.png')
gc.collect();

print("Model Evaluation Stage")
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = lgb_clf.predict(testing)
lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
lgsub['deal_probability'] = lgsub['deal_probability'].clip(0.0, 1.0) # Between 0 and 1
lgsub.to_csv("./May_20/lgsub_v4.csv.gz", index=True, header=True, compression='gzip')
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))