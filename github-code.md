# Introduction
## Intended Audience
## Libraries Used
## Getting Help

# Datasets


## Cleanup



## Cleanup Pipeline

```python
def get_education(ser):
    return (ser
            .replace({'Master’s degree': 18,
                         'Bachelor’s degree': 16,
                         'Doctoral degree': 20,
'Some college/university study without earning a bachelor’s degree': 13,
                         'Professional degree': 19,
                         'I prefer not to answer': None,
                         'No formal education past high school': 12})
                         .astype(float)
    )

def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Tweak the Kaggle survey data and return a new DataFrame.

    This function takes a Pandas DataFrame containing Kaggle 
    survey data as input and returns a new DataFrame. The 
    modifications include extracting and transforming certain 
    columns, renaming columns, and selecting a subset of columns.

    Parameters
    ----------
    df_ : pd.DataFrame
        The input DataFrame containing Kaggle survey data.

    Returns
    -------
    pd.DataFrame
        The new DataFrame with the modified and selected columns.
    """    
    major_map = {
        'Computer science (software engineering, etc.)': 'cs',
        'Engineering (non-computer focused)': 'eng',
        'Mathematics or statistics': 'stat',
    }

    return (
        df_
        .assign(
            sex=pd.col('Q1'),
            age=pd.col('Q2').str.slice(0, 2).astype(int),
            country=pd.col('Q3'),
            education=pd.col('Q4').pipe(get_education),
            major=pd.col('Q5').replace(major_map)
                .where(lambda s: s.isin(major_map.values()), 'other'),
            years_exp=(
                pd.col('Q8')
                .str.replace('+', '', regex=False)
                .str.split('-', expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                pd.col('Q9')
                .str.replace('+', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('500000', '500', regex=False)
                .str.replace(
                    'I do not wish to disclose my approximate yearly compensation',
                    '0',
                    regex=False,
                )
                .str.split('-', expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=(
                pd.col('Q16_Part_1')
                .fillna(0)
                .replace('Python', 1)
                .astype(int)
            ),
            r=(
                pd.col('Q16_Part_2')
                .fillna(0)
                .replace('R', 1)
                .astype(int)
            ),
            sql=(
                pd.col('Q16_Part_3')
                .fillna(0)
                .replace('SQL', 1)
                .astype(int)
            ),
        )
        .loc[
            :,
            ['sex', 'age', 'country', 'education', 'major', 'years_exp',
             'compensation', 'python', 'r', 'sql']
        ]
    )
```


```python
import pandas as pd 

import urllib.request
import zipfile


url = 'https://github.com/mattharrison/datasets/raw/master/data/'\
    'kaggle-survey-2018.zip'
fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'


def extract_zip(src, dst, member_name):
    """Extract a member file from a zip file and read it into a pandas 
    DataFrame.
    
    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file 
            to be read into a DataFrame.
    
    Returns:
        pandas.DataFrame: DataFrame containing the contents of the 
            member file.
    """    
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode='wb') as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw

raw = extract_zip(url, fname, member_name)        
```


```pycon
>>> print(tweak_kag(raw).dtypes)
```


```python
from feature_engine import encoding, imputation
from sklearn import base, pipeline


class TweakKagTransformer(base.BaseEstimator,
    base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing 
    Kaggle survey data as input and returns a new version of 
    the DataFrame. The modifications include extracting and 
    transforming certain columns, renaming columns, and 
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable. 
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    
    def __init__(self, ycol=None):
        self.ycol = ycol
        
    def transform(self, X):
        return tweak_kag(X)
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
```



```pycon
>>> tweaker = TweakKagTransformer() 
>>> print(tweaker.fit_transform(raw))
```


```pycon
>>> nimp = imputation.MeanMedianImputer(imputation_method='median',
...    variables=['education', 'years_exp'])
>>> print(nimp.fit_transform(tweaker
...    .fit_transform(raw)
...    .loc[:, ['education', 'years_exp']]
... ))
```


```pycon
>>> cat = encoding.OneHotEncoder(top_categories=5, drop_last=True,
...     variables=['sex', 'country', 'major'])
>>> print(cat.fit_transform(tweaker
...    .fit_transform(raw)
...    .loc[:, ['sex', 'country', 'major']]
.. ))
```


```python
def get_rawX_y(df, y_col):
    raw = (df
            .query('Q3.isin(["United States of America", "China", "India"]) '
               'and Q6.isin(["Data Scientist", "Software Engineer"])')
          )
    return raw.drop(columns=[y_col]), raw[y_col]


## Create a pipeline
kag_pl = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
     ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
           variables=['sex', 'country', 'major'])),
     ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
          variables=['education', 'years_exp']))]
)
```

## Splitting the Data






## Pipeline Methods

```pycon
>>> from sklearn import model_selection
>>> kag_X, kag_y = get_rawX_y(raw, 'Q6')
    
>>> kag_X_train, kag_X_test, kag_y_train, kag_y_test = \
...    model_selection.train_test_split(
...        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y)

>>> X_train = kag_pl.fit_transform(kag_X_train, kag_y_train)
>>> X_test = kag_pl.transform(kag_X_test)
>>> print(X_train)
```


```pycon
>>> kag_y_train
```



## Skrub

```python
from skrub import TableVectorizer

skrub_pl = pipeline.Pipeline(
    [
        ('tweak', TweakKagTransformer()),
        ('skrub', TableVectorizer(cardinality_threshold=5)),
    ]
)
```





```pycon
>>> X_train_skrub = skrub_pl.fit_transform(kag_X_train, kag_y_train)
>>> X_test_skrub = skrub_pl.transform(kag_X_test)

>>> print(X_train_skrub.iloc[:5, :8])
>>> print(X_train_skrub.shape, X_test_skrub.shape)
```


## Summary
## Exercises

# Exploratory Data Analysis


## Correlations


```python
(X_train
 .assign(data_scientist = kag_y_train == 'Data Scientist')
 .corr(method='spearman')
 .style
 .background_gradient(cmap='RdBu', vmax=1, vmin=-1)
 .set_sticky(axis='index')
)
```


## Bar Plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3))
(X_train
 .assign(data_scientist = kag_y_train)
 .groupby('r')
 .data_scientist
 .value_counts()
 .unstack()
 .plot.bar(ax=ax, title='Count of R Users (1) vs. Non-R Users (0)')
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
```




```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.crosstab(index=X_train['major_cs'], 
             columns=kag_y)
    .plot.bar(ax=ax, 
        title='Count of Computer Science Majors (1) vs. Non-CS Majors (0)')
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
(X_train
 .plot.scatter(x='years_exp', y='compensation', alpha=.3, ax=ax, c='purple',
    title='Scatter of Compensation vs. Years of Experience')
)
```



```python
import seaborn.objects as so
fig = plt.figure(figsize=(6, 3))
(so
 .Plot(X_train.assign(title=kag_y_train), x='years_exp', y='compensation', color='title')
 .add(so.Dots(alpha=.3, pointsize=2), so.Jitter(x=.5, y=10_000))
 .add(so.Line(), so.PolyFit())
 .on(fig)  # not required unless saving to image
 .plot()   # ditto
)

fig.suptitle('Scatter of Compensation vs. Years of Experience')
```



```python
fig = plt.figure(figsize=(6, 3))
(so
 .Plot(X_train
       .assign(
         title=kag_y_train,
         country=(X_train
             .loc[:, 'country_United States of America': 'country_China']
             #remove "country_" prefix from column names
             .rename(columns=lambda col: col.replace('country_', ''))
             .idxmax(axis='columns')
            )
       ), x='years_exp', y='compensation', color='title')
 .facet('country')
 .add(so.Dots(alpha=.01, pointsize=2, color='grey' ), so.Jitter(x=.5, y=10_000), col=None)
 .add(so.Dots(alpha=.5, pointsize=1.5), so.Jitter(x=.5, y=10_000))
 .add(so.Line(pointsize=1), so.PolyFit(order=2))
 .scale(x=so.Continuous().tick(at=[0,1,2,3,4,5]))
 .limit(y=(-10_000, 200_000), x=(-1, 6))  # zoom in with this not .query (above)
 .on(fig)  # not required unless saving to image
 .plot()   # ditto
)
fig.suptitle('Scatter of Compensation vs. Years of Experience by Country')
fig.subplots_adjust(top=0.80, hspace=0.3)
```


## PCA on Data

```python
from sklearn import decomposition, preprocessing, set_config

# output Pandas DataFrames from pipelines
set_config(transform_output='pandas')

pca_pipeline = pipeline.Pipeline(
    [('tweak', TweakKagTransformer()),
     ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True,
           variables=['sex', 'country', 'major'])),
     ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
           variables=['education', 'years_exp'])),
     ('std', preprocessing.StandardScaler()),
     ('pca', decomposition.PCA(n_components=3))
    ]
)

pca_pipeline.fit(kag_X_train, kag_y_train)
```

# output pandas DataFrames from pipelines


```python
pca = pca_pipeline.named_steps['pca']
```


```pycon
>>> print(pca.components_)
```



```pycon
>>> comps = pd.DataFrame(pca.components_, columns=X_train.columns,
...   index=['PC1', 'PC2', 'PC3'])
>>> print(comps)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
(comps
.plot.bar(ax=ax, title='PCA Component Weights')
.legend(bbox_to_anchor=(1,1))
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))

def fix_silly_naming(col):
    # sklearn names PC1 as pca0. (odd)
    num = col.split('pca')[-1]
    return f'PC{int(num)+1}'

embeddings = pca_pipeline.transform(kag_X_train)
(embeddings
.rename(columns=fix_silly_naming)
.plot.scatter(x='PC1', y='PC2', 
   ax=ax, title='PCA Scatter Plot',
   c=kag_y_train
   )
)

```



```python
# color by gender
fig, ax = plt.subplots(figsize=(6, 3))

(embeddings
.rename(columns=fix_silly_naming)
.plot.scatter(x='PC1', y='PC2', 
   ax=ax, title='PCA Scatter Plot',
   c=kag_X_train.Q1
   )
)

```



```python
import plotly.express as px

fig = px.scatter_3d(
    (embeddings
     .rename(columns=fix_silly_naming)
     # add original column values for hover
     .assign(**kag_X_train)
    ),
    x='PC1', y='PC2', z='PC3',
    color=kag_y_train,
    title='PCA 3D Scatter Plot',
    # show original column values on hover
    hover_data=kag_X_train.columns.tolist()
)

# shrink the size of the points
fig.update_traces(marker=dict(size=3))
# fit width of notebook
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig
```



## Summary
## Exercises

# Tree Creation
## Gini Impurity


```python
import numpy as np
import numpy.random as rn

pos_center = 12
pos_count = 100
neg_center = 7
neg_count = 1000
rs = rn.RandomState(rn.MT19937(rn.SeedSequence(42)))
X = pd.DataFrame({'value':
    np.append((pos_center) + rs.randn(pos_count),
              (neg_center) + rs.randn(neg_count))})
y = pd.Series(['pos']*pos_count + ['neg']*neg_count)


fig, ax = plt.subplots(figsize=(6, 3))
_ = (X
    .assign(label=y)
    .groupby('label')
    [['value']]
    .plot.hist(bins=30, alpha=.5, ax=ax, edgecolor='black')
)
ax.legend(['Negative', 'Positive'])
# add title to ax
ax.set_title('Value Distribution by Label')
```




```python
def calc_gini(X, y, val_col, pos_val, split_point, debug=False):
    """
    Calculate the weighted Gini impurity for a binary split.

    This function evaluates a split of the feature ``val_col`` at
    ``split_point``. Rows with values greater than or equal to the split
    go to the positive side, and the rest go to the negative side.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Target labels.
    val_col : str
        The feature column used for the split.
    pos_val : str or int
        The value in ``y`` representing the positive class.
    split_point : float
        The threshold used to split the feature.
    debug : bool, default False
        If True, print the child Gini impurities and weighted average.

    Returns
    -------
    float
        The weighted average Gini impurity of the two child nodes.
    """
    ge_split = X[val_col] >= split_point
    eq_pos = y == pos_val

    tp = (ge_split & eq_pos).sum()
    fp = (ge_split & ~eq_pos).sum()
    tn = (~ge_split & ~eq_pos).sum()
    fn = (~ge_split & eq_pos).sum()

    pos_size = tp + fp
    neg_size = tn + fn
    total_size = len(X)

    if pos_size == 0:
        gini_pos = 0
    else:
        gini_pos = 1 - (tp / pos_size) ** 2 - (fp / pos_size) ** 2

    if neg_size == 0:
        gini_neg = 0
    else:
        gini_neg = 1 - (tn / neg_size) ** 2 - (fn / neg_size) ** 2

    weighted_avg = (
        gini_pos * (pos_size / total_size) +
        gini_neg * (neg_size / total_size)
    )

    if debug:
        print(f"{gini_pos=:.3f} {gini_neg=:.3f} {weighted_avg=:.3f}")

    return weighted_avg
```


```pycon
>>> calc_gini(X, y, val_col='value', pos_val='pos', split_point=9.24, debug=True)
```


```python
values = np.arange(5, 15, .1)
ginis = []
for v in values:
    ginis.append(calc_gini(X, y, val_col='value',   
                           pos_val='pos', split_point=v))
fig, ax = plt.subplots(figsize=(6, 3))    
ax.plot(values, ginis)
ax.set_title('Gini Impurity for Different Split Points')
ax.set_ylabel('Gini Impurity')
ax.set_xlabel('Split Point')
```



```pycon
>>> pd.Series(ginis, index=values).loc[9.5:10.5]
```


```pycon
>>> print(pd.DataFrame({'gini':ginis, 'split':values})
...  .query('gini <= gini.min()')
... )
```

## Coefficients in Trees

```python
from sklearn import tree
stump = tree.DecisionTreeClassifier(max_depth=1)
stump.fit(X, y)
```


```pycon
>>> gini_pos = 0.039
>>> gini_neg = 0.002
>>> pos_size = 101
>>> neg_size = 999
>>> total_size = pos_size + neg_size
>>> weighted_avg = gini_pos * (pos_size/total_size) + \
...               gini_neg * (neg_size/total_size)
>>> print(weighted_avg)
```

## How XGBoost Chooses a Split


## Leaf Values in XGBoost


## Gradients and Hessians on Real Data

```python
import xgboost as xgb

xg_small = xgb.XGBClassifier(n_estimators=1, max_depth=1)
xg_small.fit(X, y=='pos')

probs = xg_small.predict_proba(X)

scores = (X
    .assign(y=(y=='pos').astype(int),
            p=probs[:,1],
            g=pd.col('p') - (pd.col('y')),
            h=pd.col('p') * (1 - pd.col('p'))
)
 )
```


```pycon
>>> print(scores.query('value < 10'))
```


## Aggregating Gradients and Hessians Inside a Candidate Split

```python
split_mask = X['value'] < 9.85
left = scores[split_mask]
right = scores[~split_mask]

summary = pd.DataFrame({
    'leaf': ['left', 'right'],
    'G': [left.g.sum(), right.g.sum()],
    'H': [left.h.sum(), right.h.sum()],
    'count': [len(left), len(right)]
})
```

```pycon
>>> print(summary)
```


```pycon
>>> reg_lambda = 1
>>> print(summary
...      .assign(
...          leaf_value=-pd.col('G') / (pd.col('H') + reg_lambda)
...      ))
```


## Looking at a Few Real Rows

```pycon
>>> print(left)
```

## Visualizing the Row-Level Gradients

```python
import highlight_text as ht

fig, ax = plt.subplots(figsize=(6, 3))
colors = scores.y.map({0:'red', 1:'blue'})

(scores
 .plot.scatter(x='value', y='g', alpha=.5, c=colors, ax=ax)
)
split_value = 9.85
# add title with highlighted text
ht.fig_text(0.1, .95, f"Gradient (g) vs Value with Split at {split_value}\n<Negative> <Positive>", 
            fontsize=14,
            ax=ax, ha='center', va='bottom',
           highlight_textprops=[{"color": "red", "fontsize":12}, 
                                {"color": "blue", "fontsize":12}])
```

## Checking the Gain of the Split

```python
g = scores.g.sum()
h = scores.h.sum()

g_l = left.g.sum()
h_l = left.h.sum()

g_r = right.g.sum()
h_r = right.h.sum()

reg_lambda = 1
gamma = 0

gain = 0.5 * (g_l**2 / (h_l + reg_lambda) 
            + g_r**2 / (h_r + reg_lambda) 
            - g**2 / (h + reg_lambda)) - gamma
```


```pycon
>>> print(gain)
```


## The Default Classification Objective



## Changing the Objective

## Default Evaluation Metric


## Missing Values and Default Directions

## Summary
## Exercises

# Tree Visualization


## `plot_tree` with scikit-learn

```python
fig, ax = plt.subplots(figsize=(6, 6))
tree.plot_tree(stump, feature_names=['value'],
               filled=True, 
               class_names=stump.classes_,
               ax=ax)
```


## `plot_tree` and Graphviz with XGBoost

```python
import xgboost as xgb
xg_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
xg_stump.fit(X[['value']], (y == 'pos').astype(int))
```


```python
fig, ax = plt.subplots(figsize=(6, 4))
xgb.plot_tree(xg_stump, num_trees=0, ax=ax)
```




```python
import subprocess
def my_dot_export(xg, num_tree, filename, title='', direction='TB'):
    """Exports a specified number of trees from an XGBoost model as a graph 
    visualization in dot and png formats.

    Args:
        xg: An XGBoost model.
        num_tree: The number of tree to export.
        filename: The name of the file to save the exported visualization.
        title: The title to display on the graph visualization (optional).
        direction: The direction to lay out the graph, either 'TB' (top to 
            bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(xg, num_trees=num_tree)
    content = f'''    node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    '''
    out = res.source.replace('graph [ rankdir=TB ]', 
                             f'graph [ rankdir={direction} ];\n {content}')
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot 
    dot_filename = filename
    with open(dot_filename, 'w') as fout:
        fout.write(out)
    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}'.split())
```


```python
my_dot_export(xg_stump, num_tree=0, filename='img/stump_xg.dot', title='A demo stump')    
```



## `dtreeviz`

```python
import dtreeviz
viz = dtreeviz.model(
    xg_stump,
    X_train=X[['value']],
    y_train=(y == 'pos').astype(int),
    target_name='positive',
    feature_names=['value'],
    class_names=['negative', 'positive'],
    tree_index=0,
)
viz.view()
```



## `supertree`

```python
from supertree import SuperTree

supertree_stump = SuperTree(
    stump,
    X[['value']],
    y,
    feature_names=['value'],
    target_names=['neg', 'pos'],
)

supertree_stump.show_tree(start_depth=2)
```



```python
supertree_stump.save_html('img/supertree_stump.html', start_depth=2)
```


```python
xg_default = xgb.XGBClassifier()
xg_default.fit(X[['value']], (y == 'pos').astype(int))
supertree_xgb = SuperTree(
    xg_default,
    X[['value']],
    (y == 'pos').astype(int),
    feature_names=['value'],
    target_names=['negative', 'positive'],
)

supertree_xgb.show_tree(which_tree=0, start_depth=2, max_samples=2_000)
```



## Summary
## Exercises

# Stumps on Real Data
## Scikit-learn 

```python
stump_dt = tree.DecisionTreeClassifier(max_depth=1)
X_train = kag_pl.fit_transform(kag_X_train)
stump_dt.fit(X_train, kag_y_train)
```




```python
fig, ax = plt.subplots(figsize=(6, 6))
features = list(c for c in X_train.columns)
tree.plot_tree(stump_dt, feature_names=features, 
               filled=True, 
               class_names=stump_dt.classes_,
               ax=ax)
```




```pycon
>>> X_test = kag_pl.transform(kag_X_test)
>>> stump_dt.score(X_test, kag_y_test)
```



```pycon
>>> from sklearn import dummy
>>> dummy_model = dummy.DummyClassifier()
>>> dummy_model.fit(X_train, kag_y_train)
>>> dummy_model.score(X_test, kag_y_test)
```


## Decision Stump with XGBoost


```pycon
>>> import xgboost as xgb
>>> kag_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
>>> kag_stump.fit(X_train, kag_y_train)
```


```pycon
>>> print(kag_y_train)
```


```pycon
>>> print(kag_y_train == 'Software Engineer')
```



```pycon
>>> from sklearn import preprocessing
>>> label_encoder = preprocessing.LabelEncoder()
>>> y_train = label_encoder.fit_transform(kag_y_train)
>>> y_test = label_encoder.transform(kag_y_test)
>>> y_test[:5]
```


```pycon
>>> label_encoder.classes_
```


```pycon
>>> label_encoder.inverse_transform([0, 1])
```



```pycon
>>> kag_stump = xgb.XGBClassifier(n_estimators=1, max_depth=1)
>>> kag_stump.fit(X_train, y_train)
>>> kag_stump.score(X_test, y_test)
```


```python
my_dot_export(kag_stump, num_tree=0, filename='img/stump_xg_kag.dot', 
              title='XGBoost Stump')    
```


## Values in the XGBoost Tree

```pycon
>>> kag_stump.classes_
```

```python
import numpy as np
def inv_logit(p: float) -> float:
    """
    Compute the inverse logit function of a given value.

    The inverse logit function is defined as:
        f(p) = exp(p) / (1 + exp(p))

    Parameters
    ----------
    p : float
        The input value to the inverse logit function.

    Returns
    -------
    float
        The output of the inverse logit function.
    """
    return np.exp(p) / (1 + np.exp(p))
```


```pycon
>>> inv_logit(.0717741922)
```

```python
inv_logit(-2.19)
```



```pycon
>>> inv_logit(-.3592)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
vals = np.linspace(-7, 7)
ax.plot(vals, inv_logit(vals))
ax.set_title('Inverse Logit Function')
# background color for positive and negative regions
ax.axvspan(0, 7, color='blue', alpha=.1)
ax.axvspan(-7, 0, color='green', alpha=.1)
ax.set_xlim(-7, 7)
ax.annotate('Crossover point', (0,.5), (-5,.8), arrowprops={'color':'k'}) 
ax.annotate('Predict Positive', (5,.6), (1,.6), va='center', arrowprops={'color':'k'}) 
ax.annotate('Predict Negative', (-5,.4), (-3,.4), va='center', arrowprops={'color':'k'}) 
```


## Summary
## Exercises

# Model Complexity & Hyperparameters
## Underfit



```pycon
>>> underfit = tree.DecisionTreeClassifier(max_depth=1)
>>> X_train = kag_pl.fit_transform(kag_X_train)
>>> underfit.fit(X_train, kag_y_train)
>>> underfit.score(X_test, kag_y_test)
```


## Growing a Tree



## Overfitting
## Overfitting with Decision Trees

```pycon
>>> hi_variance = tree.DecisionTreeClassifier(max_depth=None)
>>> X_train = kag_pl.fit_transform(kag_X_train)
>>> hi_variance.fit(X_train, kag_y_train)
>>> hi_variance.score(X_test, kag_y_test)
```


```pycon
>>> hi_variance_xg = xgb.XGBClassifier(
...  n_estimators=1, max_depth=0, min_child_weight=0, reg_lambda=0)
>>> hi_variance_xg.fit(X_train, kag_y_train == 'Data Scientist')
>>> hi_variance_xg.score(X_test, kag_y_test == 'Data Scientist')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
features = list(c for c in X_train.columns)
tree.plot_tree(hi_variance, feature_names=features, filled=True)
```



```python
fig, ax = plt.subplots(figsize=(6, 4))
features = list(c for c in X_train.columns)
tree.plot_tree(hi_variance, feature_names=features, filled=True, 
                  class_names=hi_variance.classes_,
                max_depth=2, fontsize=6)
```


## Summary
## Exercises

# Tree Hyperparameters
## Decision Tree Hyperparameters

```pycon
>>> stump.get_params()
```


## Validation Curves


```python
accuracies = []
for depth in range(1, 15):
    between = tree.DecisionTreeClassifier(max_depth=depth)
    between.fit(X_train, kag_y_train)
    accuracies.append(between.score(X_test, kag_y_test))
fig, ax = plt.subplots(figsize=(6, 3))    
(pd.Series(accuracies, name='Accuracy', index=range(1, len(accuracies)+1))
 .plot(ax=ax, title='Accuracy at a given Tree Depth'))
ax.set_ylabel('Accuracy')
ax.set_xlabel('max_depth')
```



```pycon
>>> between = tree.DecisionTreeClassifier(max_depth=7)
>>> between.fit(X_train, kag_y_train)
>>> between.score(X_test, kag_y_test)
```

## Better Validation Curves with Yellowbrick

```python
from yellowbrick.model_selection import validation_curve
fig, ax = plt.subplots(figsize=(6, 3))    
viz = validation_curve(tree.DecisionTreeClassifier(),
    X=pd.concat([X_train, X_test]),
    y=pd.concat([kag_y_train, kag_y_test]),   
    param_name='max_depth', param_range=range(1,14),
    scoring='accuracy', cv=5, ax=ax, n_jobs=6)                           
```



## Grid Search


```python
from sklearn.model_selection import GridSearchCV
params = {
    'max_depth': [3, 5, 7, 8],
    'min_samples_leaf': [1, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 4, 5, 6],
}
grid_search = GridSearchCV(estimator=tree.DecisionTreeClassifier(), 
                           param_grid=params, cv=4, n_jobs=-1, 
                           verbose=1, scoring="accuracy")
grid_search.fit(X_train, kag_y_train)
```



```pycon
>>> grid_search.best_params_
```


```pycon
>>> between2 = tree.DecisionTreeClassifier(**grid_search.best_params_)
>>> between2.fit(X_train, kag_y_train)
>>> between2.score(X_test, kag_y_test)
```



```python
# why is the score different than between_tree?
(pd.DataFrame(grid_search.cv_results_)
 .sort_values(by='rank_test_score')
 .style
 .background_gradient(axis='rows')
)
```




```pycon
>>> results = model_selection.cross_val_score(
...    tree.DecisionTreeClassifier(max_depth=7),
...    X=pd.concat([X_train, X_test], axis='index'),
...    y=pd.concat([kag_y_train, kag_y_test], axis='index'),
...    cv=4
... )

>>> results
```

```pycon
>>> results.mean()
```


```pycon
>>> results = model_selection.cross_val_score(
...    tree.DecisionTreeClassifier(max_depth=7, min_samples_leaf=5,
...                                min_samples_split=2),
...    X=pd.concat([X_train, X_test], axis='index'),
...    y=pd.concat([kag_y_train, kag_y_test], axis='index'),
...    cv=4
... )

>>> results
```

```pycon
>>> results.mean()
```


## Summary
## Exercises

# Random Forest
## Ensembles with Bagging
## Scikit-learn Random Forest

```pycon
>>> from sklearn import ensemble
>>> rf = ensemble.RandomForestClassifier(random_state=42)
>>> rf.fit(X_train, kag_y_train)
>>> rf.score(X_test, kag_y_test)
```



```pycon
>>> rf.get_params()
```



```pycon
>>> len(rf.estimators_)
```


```pycon
>>> print(rf.estimators_[0])
```


```python

fig, ax = plt.subplots(figsize=(6, 3.95))
features = list(c for c in X_train.columns)
tree.plot_tree(rf.estimators_[0], feature_names=features, 
               filled=True, class_names=rf.classes_, ax=ax,
               max_depth=2, fontsize=6)
```



## XGBoost Random Forest

```pycon
>>> import xgboost as xgb
>>> rf_xg = xgb.XGBRFClassifier(random_state=42)
>>> rf_xg.fit(X_train, y_train) 
>>> rf_xg.score(X_test, y_test)
```



```pycon
>>> rf_xg.get_params()
```


```python
fig, ax = plt.subplots(figsize=(6,12), dpi=600)
xgb.plot_tree(rf_xg, num_trees=0, ax=ax, size='1,1')
```

```python
my_dot_export(rf_xg, num_trees=0, filename='img/rf_xg_kag.dot', 
              title='First Random Forest Tree', direction='LR')    
```



```python
viz = dtreeviz.model(rf_xg, X_train=X_train,
    y_train=y_train,
    target_name='Job', feature_names=list(X_train.columns), 
    class_names=['DS', 'SE'], tree_index=0)
viz.view(depth_range_to_display=[0,2])
```


## Random Forest Hyperparameters

## Training the Number of Trees in the Forest

```python
from yellowbrick.model_selection import validation_curve
fig, ax = plt.subplots(figsize=(6, 3))    
viz = validation_curve(xgb.XGBRFClassifier(random_state=42),
    X=pd.concat([X_train, X_test], axis='index'),
    y=np.concatenate([y_train, y_test]),
    param_name='n_estimators', param_range=range(1, 100, 2),
    scoring='accuracy', cv=3, 
    ax=ax)                           
```




```pycon
>>> rf_xg43 = xgb.XGBRFClassifier(random_state=42, n_estimators=43)
>>> rf_xg43.fit(X_train, y_train) 
>>> rf_xg43.score(X_test, y_test)
```

## Summary
## Exercises

# XGBoost


## Jargon

## Benefits of Boosting



## Creating an XGBoost Model

```python
from sklearn import base, compose, datasets, ensemble, \
    metrics, model_selection, pipeline, preprocessing, tree
import xgboost as xgb

import xg_helpers as xhelp
```

```python
url = 'https://github.com/mattharrison/datasets/raw/master/data/'\
    'kaggle-survey-2018.zip'
fname = 'kaggle-survey-2018.zip'
member_name = 'multipleChoiceResponses.csv'

raw = xhelp.extract_zip(url, fname, member_name)
```


```python
## Create raw X and raw y
kag_X, kag_y = xhelp.get_rawX_y(raw, 'Q6')
    
## Split data    
kag_X_train, kag_X_test, kag_y_train, kag_y_test = \
    model_selection.train_test_split(
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y)    

## Transform X with pipeline
X_train = xhelp.kag_pl.fit_transform(kag_X_train)
X_test = xhelp.kag_pl.transform(kag_X_test)

## Transform y with label encoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(kag_y_train)
y_train = label_encoder.transform(kag_y_train)
y_test = label_encoder.transform(kag_y_test)

# Combined Data for cross validation/etc
X = pd.concat([X_train, X_test], axis='index')
y = pd.Series([*y_train, *y_test], index=X.index)
```


## A Boosted Model

```pycon
>>> xg_oob = xgb.XGBClassifier()
>>> xg_oob.fit(X_train, y_train)
>>> xg_oob.score(X_test, y_test)
```


```pycon
>>> # Let's try w/ depth of 2 and 2 trees
>>> xg2 = xgb.XGBClassifier(max_depth=2, n_estimators=2)
>>> xg2.fit(X_train, y_train)
>>> xg2.score(X_test, y_test)
```



```python
import dtreeviz

viz = dtreeviz.model(xg2, X_train=X, y_train=y, target_name='Job',
    feature_names=list(X_train.columns), 
    class_names=['DS', 'SE'], tree_index=0)
viz.view(depth_range_to_display=[0,2])
```


## Understanding the Output of the Trees

```python
xhelp.my_dot_export(xg2, num_trees=0, filename='img/xgb_md2.dot', 
                    title='First Tree') 
```







```pycon
>>> # Predicts 0 - Data Scientist
>>> se7894 = pd.DataFrame({'age': {7894: 22},                                            
... 'education': {7894: 16.0},
... 'years_exp': {7894: 1.0},
... 'compensation': {7894: 0},
... 'python': {7894: 1},
... 'r': {7894: 0},
... 'sql': {7894: 0},
... 'sex_Male': {7894: 1},                                   
... 'sex_Female': {7894: 0},
... 'sex_Prefer not to say': {7894: 0},
... 'sex_Prefer to self-describe': {7894: 0},
... 'country_United States of America': {7894: 0},
... 'country_India': {7894: 1},
... 'country_China': {7894: 0},
... 'major_cs': {7894: 0},
... 'major_other': {7894: 0},
... 'major_eng': {7894: 0},
... 'major_stat': {7894: 0}})
>>> xg2.predict_proba(se7894)
```


```pycon
>>> # Predicts 0 - Data Scientist
>>> xg2.predict(pd.DataFrame(se7894))
```


```python
xhelp.my_dot_export(xg2, num_trees=1, filename='img/xgb_md2_tree1.dot', title='Second Tree') 
```




```python
def inv_logit(p):
    "Convert log-odds to probability"
    return np.exp(p) / (1 + np.exp(p))

def logit(p):
    "Convert probability to log-odds"
    return np.log(p / (1 - p))
```


```pycon
>>> xg2.base_score is None
```


```pycon
>>> import json
>>> booster = xg2.get_booster()
>>> cfg = booster.save_config()
>>> json.loads(cfg)
```


```pycon
>>> json.loads(cfg)['learner']['learner_model_param']['base_score']
```



```pycon
>>> base_score = .45355
>>> log_odds_base = logit(base_score)
>>> log_odds_base
```


```pycon
>>> log_odds_tree1 = 0.1289
>>> log_odds_tree2 = -0.02959
>>> total = log_odds_base +  log_odds_tree1 + log_odds_tree2
>>> inv_logit(total)
```



```pycon
>>> xg2.predict_proba(se7894)
```



## Summary
## Exercises

# Early Stopping
## Early Stopping Rounds

```pycon
>>> # Defaults
>>> xg = xgb.XGBClassifier()
>>> xg.fit(X_train, y_train)
>>> xg.score(X_test, y_test)
```


```pycon
>>> xg = xgb.XGBClassifier(early_stopping_rounds=20, random_state=42)
>>> xg.fit(X_train, y_train,
...        eval_set=[(X_train, y_train),
...                  (X_test, y_test)
...                 ]
...       )
>>> xg.score(X_test, y_test)
```


```pycon
>>> xg.best_iteration
```


```pycon
>>> xg = xgb.XGBClassifier(early_stopping_rounds=20, random_state=42)
>>> xg.fit(X_train, y_train,
...        eval_set=[(X_train, y_train),
...                  (X_test, y_test)
...                 ], 
...        verbose=False
...       )
>>> xg.score(X_test, y_test)
```


## Plotting Tree Performance

```pycon
>>> # validation_0 is for training data
>>> # validation_1 is for testing data
>>> results = xg.evals_result()
>>> results
```


```python
# Testing score is best at 13 trees
results = xg.evals_result()
fig, ax = plt.subplots(figsize=(6, 3))
ax = (pd.DataFrame({'training': results['validation_0']['logloss'],
                    'testing': results['validation_1']['logloss']})
  .assign(ntrees=lambda adf: range(1, len(adf)+1))      
  .set_index('ntrees')
  .plot(ax=ax, 
        title='eval_results with early_stopping')
)
ax.annotate(f'Best number \nof trees ({xg.best_iteration})', 
      xy=(xg.best_iteration, results['validation_1']['logloss'][xg.best_iteration-1]),
      xytext=(xg.best_iteration+7, results['validation_1']['logloss'][xg.best_iteration-1]-0.08), 
      arrowprops={'color':'k'})
ax.set_xlabel('ntrees')
ax.set_ylabel('logloss')
```



```python
# Using value from early stopping gives same result
>>> xg13 = xgb.XGBClassifier(n_estimators=13)
>>> xg13.fit(X_train, y_train,
...          eval_set=[(X_train, y_train),
...                    (X_test, y_test)]
... )
```

```pycon
>>> xg13.score(X_test, y_test)
```


```pycon
>>> xg.score(X_test, y_test)
```


```pycon
>>> # No early stopping, uses all estimators
>>> xg_no_es = xgb.XGBClassifier()
>>> xg_no_es.fit(X_train, y_train)
>>> xg_no_es.score(X_test, y_test)
```


## Different `eval_metrics`


```pycon
>>> xg_err = xgb.XGBClassifier(early_stopping_rounds=20, 
...                            eval_metric='error')
>>> xg_err.fit(X_train, y_train,
...        eval_set=[(X_train, y_train),
...                  (X_test, y_test)
...                 ]
...       )
>>> xg_err.score(X_test, y_test)
```


```pycon
>>> xg_err.best_iteration
```


## Summary
## Exercises

# XGBoost Hyperparameters
## Hyperparameters

# Example: Increasing weight for specific rows

## Examining Hyperparameters

```pycon
>>> xg = xgb.XGBClassifier() # set the hyperparameters in here
>>> xg.fit(X_train, y_train)
>>> xg.get_params()
```


## Tuning Hyperparameters


```python
fig, ax = plt.subplots(figsize=(6, 3))
ms.validation_curve(xgb.XGBClassifier(), X_train, y_train, param_name='gamma', 
    param_range=[0, .5, 1,5,10, 20, 30], n_jobs=-1, ax=ax)
```



## Intuitive Understanding of Learning Rate

```python
# check impact of learning weight on scores
xg_lr1 = xgb.XGBClassifier(learning_rate=1, max_depth=2)
xg_lr1.fit(X_train, y_train)
```

```python
my_dot_export(xg_lr1, num_trees=0, filename='img/xg_depth2_tree0.dot', 
              title='Learning Rate set to 1')    
```




```python
# check impact of learning weight on scores
xg_lr001 = xgb.XGBClassifier(learning_rate=.001, max_depth=2)
xg_lr001.fit(X_train, y_train)
```

```python
my_dot_export(xg_lr001, num_trees=0, filename='img/xg_depth2_tree0_lr001.dot',
              title='Learning Rate set to .001')    
```



## Grid Search



```python
from sklearn import model_selection
params = {'reg_lambda': [0],  # No effect
          'learning_rate': [.1, .3], # makes each boost more conservative 
          'subsample': [.7, 1],
          'max_depth': [2, 3],
          'random_state': [42],
          'n_jobs': [-1],
          'n_estimators': [200]}

xgb2 = xgb.XGBClassifier(early_stopping_rounds=5) 
cv = (model_selection.GridSearchCV(xgb2, params, cv=3, n_jobs=-1)
    .fit(X_train, y_train,
         eval_set=[(X_test, y_test)],
         verbose=50
    )
)
```


```pycon
>>> cv.best_params_
```


```python
params = {'learning_rate': 0.3,
 'max_depth': 3,
 'n_estimators': 200,
 'n_jobs': -1,
 'random_state': 42,
 'reg_lambda': 0,
 'subsample': 0.7}

xgb_grid = xgb.XGBClassifier(**params, early_stopping_rounds=50)
xgb_grid.fit(X_train, y_train, eval_set=[(X_train, y_train),
 (X_test, y_test)],
 verbose=10
)
```


```python
# vs default
xgb_def = xgb.XGBClassifier(early_stopping_rounds=50)
xgb_def.fit(X_train, y_train, eval_set=[(X_train, y_train),
 (X_test, y_test)],
 verbose=10
)
```


```pycon
>>> xgb_def.score(X_test, y_test), xgb_grid.score(X_test, y_test)
```



```python
results_default = model_selection.cross_val_score(
    xgb.XGBClassifier(),
    X=X, y=y,
    cv=4
)
```


```pycon
>>> results_default
```


```pycon
>>> results_default.mean()
```


```python
results_grid = model_selection.cross_val_score(
    xgb.XGBClassifier(**params),
    X=X, y=y,
    cv=4
)
```


```pycon
>>> results_grid
```


```pycon
>>> results_grid.mean()
```

## Summary
## Exercises

# Optuna
## Bayesian Optimization
## Exhaustive Tuning with Optuna

```python
import optuna
from sklearn.metrics import accuracy_score, roc_auc_score

from typing import Callable

def hyperparameter_tuning(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    early_stopping_rounds: int = 50,
    metric: Callable = accuracy_score,
) -> float:
    """
    Perform hyperparameter tuning for an XGBoost classifier.

    This function takes an Optuna trial, training and test data,
    and an optional value for early stopping rounds, and returns
    the loss resulting from the tuning process. The model is trained
    using the training data and evaluated on the test data. The
    loss is computed as the negative of the accuracy score.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to suggest hyperparameters.
    X_train : pd.DataFrame
        The training data.
    y_train : pd.Series
        The training target.
    X_test : pd.DataFrame
        The test data.
    y_test : pd.Series
        The test target.
    early_stopping_rounds : int, optional
        The number of early stopping rounds to use. The default value
        is 50.
    metric : callable
        Metric to maximize. Default is accuracy

    Returns
    -------
    float
        The loss resulting from the tuning process.
    """
    space = {
        'max_depth': trial.suggest_int('max_depth', 1, 8),
        'min_child_weight': trial.suggest_float(
            'min_child_weight', 0.135335283, 20.085536923, log=True
        ),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
        'gamma': trial.suggest_float(
            'gamma', 0.00004539993, 22026.4657948, log=True
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.00091188197, 1.0, log=True
        ),
        'random_state': 42,
        'early_stopping_rounds': early_stopping_rounds,
    }

    model = xgb.XGBClassifier(**space)
    evaluation = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        verbose=False,
    )

    pred = model.predict(X_test)
    score = metric(y_test, pred)
    return -score

```


```python
sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
)

study.optimize(
    lambda trial: hyperparameter_tuning(
        trial, X_train, y_train, X_test, y_test
    ),
    n_trials=2_000,
    # timeout=60*5 # 5 minutes
)
```


```python
storage = "sqlite:///optuna_xgb.db"
saved_study = optuna.create_study(
      study_name="xgb_tuning",
      direction=study.direction,
      storage=storage,
      load_if_exists=True,
  )

saved_study.add_trials(study.trials)
```


```python
study2 = optuna.load_study(
      study_name="xgb_tuning",
      storage="sqlite:///optuna_xgb.db",
  )

```


```pycon
>>> print(study.best_params)
```


```python
# 2 hours of training (paste best in here)
long_params = {'max_depth': 4, 
    'min_child_weight': 1.5445507523999649, 
    'subsample': 0.9450931487248706, 
    'colsample_bytree': 0.6283562512266935, 
    'reg_alpha': 1.565568537751338, 
    'reg_lambda': 5.672886997758711, 
    'gamma': 0.007930574424864905, 
    'learning_rate': 0.2598078142933787}

```


```python
xg_ex = xgb.XGBClassifier(
    **long_params,
    n_estimators=1000,
    early_stopping_rounds=50,
)
xg_ex.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100,
)

```


```pycon
>>> xg_ex.score(X_test, y_test)

```

## Defining Parameter Distributions

```pycon
>>> import optuna
>>> trial = optuna.trial.FixedTrial({'value': 'b'})
>>> trial.suggest_categorical('value', ['a', 'b', 'c'])

```


```pycon
>>> trial = optuna.trial.FixedTrial({'value': 0.78})
>>> trial.suggest_float('value', 0, 1)

```


```python
import numpy as np

uniform_vals = np.random.uniform(0, 1, size=10_000)

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(uniform_vals)
ax.set_title('Uniform Distribution between 0 and 1')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
```


```python
loguniform_vals = np.exp(np.random.uniform(-5, 5, size=10_000))
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(loguniform_vals, bins=30)
ax.set_title('Log-Uniform Distribution between exp(-5) and exp(5)')
ax.set_xlabel('Value')


```


```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(np.arange(-5, 5, step=.1))
 .rename('x')
 .to_frame()
 .assign(y=lambda adf: np.exp(adf.x))
 .plot(x='x', y='y', ax=ax, title='Exponential Transform of loguniform')
)
ax.set_ylabel('exp(x)')
ax.set_xlabel('x')

```


```pycon
>>> trial = optuna.trial.FixedTrial({'value': 3.0})
>>> trial.suggest_float('value', 0.1, 10, log=True)

```


```pycon
>>> import optuna
>>> trial = optuna.trial.FixedTrial({'value': 3})
>>> trial.suggest_int('value', -5, 5, step=2)
```

## Exploring the Trials

```python
opt2hr = study.trials_dataframe()
```


```pycon
>>> print(opt2hr)
```


```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(opt2hr.select_dtypes('number').corr(method='spearman'),
    cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax,
    
)
ax.set_title('Spearman Correlation of Hyperparameters')

```


```python
fig, ax = plt.subplots(figsize=(6, 3))
opt2hr.plot.scatter(x='number', y='value', alpha=.1, color='purple', ax=ax)
ax.set_title('Training Number vs Score')
ax.set_ylabel('Loss')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
opt2hr.plot.scatter(x='params_max_depth', y='value', alpha=.2, 
                    color='purple', ax=ax)
ax.set_title('Loss vs max_depth')
ax.set_ylabel('Loss')

```


```python
import numpy as np

def jitter(df: pd.DataFrame, col: str, amount: float = 1) -> pd.Series:
    """
    Add random noise to the values in a Pandas DataFrame column.

    This function adds random noise to the values in a specified
    column of a Pandas DataFrame. The noise is uniform random
    noise with a range of `amount` centered around zero. The
    function returns a Pandas Series with the jittered values.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    col : str
        The name of the column to jitter.
    amount : float, optional
        The range of the noise to add. The default value is 1.

    Returns
    -------
    pd.Series
        A Pandas Series with the jittered values.
    """
    vals = np.random.uniform(
        low=-amount / 2,
        high=amount / 2,
        size=df.shape[0],
    )
    return df[col] + vals

```


```python
fig, ax = plt.subplots(figsize=(6,3))
(
    opt2hr
    .assign(max_depth=lambda df: jitter(df, 'params_max_depth', amount=.8))
    .plot.scatter(x='max_depth', y='value', alpha=.1, color='purple', ax=ax)
)

ax.set_title('Loss vs max_depth (jittered)')
ax.set_ylabel('Loss')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
(
    opt2hr
    .assign(max_depth=lambda df: jitter(df, 'params_max_depth', amount=.8))
    .plot.scatter(
        x='max_depth',
        y='value',
        alpha=.5,
        c='number',
        cmap='viridis',
        ax=ax,
    )
)

ax.set_title('Loss vs max_depth (jittered) with Training Number')
ax.set_ylabel('Loss')

# annotate min loss
ax.annotate(f'Min Loss ({opt2hr.value.min():.3f}) \n Step {opt2hr.number[opt2hr.value.idxmin()]}', 
      xy=(opt2hr.params_max_depth[opt2hr.value.idxmin()], opt2hr.value.min()),
      xytext=(opt2hr.params_max_depth[opt2hr.value.idxmin()]+.5, opt2hr.value.min()+0.05), 
      arrowprops={'color':'k'})
```


```python
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 3))
sns.violinplot(x='params_max_depth', y='value', data=opt2hr, ax=ax)
ax.set_title('Distribution of Loss at each max_depth')
ax.set_ylabel('Loss')

```


```python
fig, ax = plt.subplots(figsize=(6, 3))
opt2hr.plot.scatter(x='params_gamma', y='value', alpha=.1, color='purple', ax=ax,
                    #logx=True
                    )
# set x values to not use scientific notation
#ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1_000])
#ax.set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100, 1_000])
ax.set_title('Loss vs gamma')
ax.set_ylabel('Loss')

```


```python
fig, ax = plt.subplots(figsize=(6, 3))
opt2hr.plot.scatter(
    x='params_gamma',
    y='value',
    alpha=.5,
    c='number',
    cmap='viridis',
    ax=ax,
    logx=True,
)

ax.set_xticks([0.001, 0.01, 0.1, 1, 10, 100, 1_000])
ax.set_xticklabels([0.001, 0.01, 0.1, 1, 10, 100, 1_000])
ax.set_title('Loss vs gamma (log scale) with Training Number')
# annotate min loss
ax.set_ylabel('Loss')
ax.annotate(f'Min Loss ({opt2hr.value.min():.3f}) \n Step {opt2hr.number[opt2hr.value.idxmin()]}', 
      xy=(opt2hr.params_gamma[opt2hr.value.idxmin()], opt2hr.value.min()),
      xytext=(opt2hr.params_gamma[opt2hr.value.idxmin()]*1.5, opt2hr.value.min()+0.05), 
      arrowprops={'color':'k'})
```

### EDA with Plotly

```python
import plotly.graph_objects as go

def plot_3d_mesh(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> go.Figure:
    """
    Create a 3D mesh plot using Plotly.

    This function creates a 3D mesh plot using Plotly, with
    the `x_col`, `y_col`, and `z_col` columns of the `df`
    DataFrame as the x, y, and z values, respectively.
    The function returns a Plotly Figure object.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        The name of the column to use as the x values.
    y_col : str
        The name of the column to use as the y values.
    z_col : str
        The name of the column to use as the z values.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the 3D mesh plot.
    """
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                intensity=df[z_col] / df[z_col].min(),
                hovertemplate=(
                    f"{z_col}: %{{z}}<br>{x_col}: %{{x}}<br>{y_col}: "
                    "%{{y}}<extra></extra>"
                ),
            )
        ],
    )

    fig.update_layout(
        title=dict(text=f'{y_col} vs {x_col}'),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=50),
    )
    return fig

```



```python
fig = plot_3d_mesh(
    opt2hr.query('params_gamma < .2'),
    'params_reg_lambda',
    'params_gamma',
    'value',
)

fig

```


```python
import plotly.express as px
import plotly.graph_objects as go

def plot_3d_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    color_col: str,
    opacity: float = 1,
) -> go.Figure:
    """
    Create a 3D scatter plot using Plotly Express.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to plot.
    x_col : str
        The name of the column to use as the x values.
    y_col : str
        The name of the column to use as the y values.
    z_col : str
        The name of the column to use as the z values.
    color_col : str
        The name of the column to use for coloring.
    opacity : float
        The opacity (alpha) of the points.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the 3D scatter plot.
    """
    fig = px.scatter_3d(
        data_frame=df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        color_continuous_scale=px.colors.sequential.Viridis_r,
        opacity=opacity,
    )
    return fig
```


```python
plot_3d_scatter(
    opt2hr.query('params_gamma < .2'),
    'params_reg_lambda',
    'params_gamma',
    'number',
    color_col='value',
)

```

### Conclusion
## Exercises

# Step-wise Tuning with sk-stepwise

## Validation Set

```python
## Create raw X and raw y
step_kag_X, step_kag_y = xhelp.get_rawX_y(raw, 'Q6')

## Hold out the final test set first: 20% of the full dataset
step_X_train_full, step_X_test, step_y_train_full, step_y_test = \
    model_selection.train_test_split(
    step_kag_X,
    step_kag_y,
    test_size=.20,
    random_state=42,
    stratify=step_kag_y,
)

## Split the remaining 80% into 60% train and 20% validation.
## .25 of the remaining .80 is .20 of the original dataset.
step_X_train, step_X_val, step_y_train, step_y_val = model_selection.train_test_split(
    step_X_train_full,
    step_y_train_full,
    test_size=.25,
    random_state=42,
    stratify=step_y_train_full,
)
```


```pycon
>>> print(step_X_train.shape, step_X_val.shape, step_X_test.shape)
```


```python
## Transform X with the pipeline fit on training rows only
step_X_train = xhelp.kag_pl.fit_transform(step_X_train)
step_X_val = xhelp.kag_pl.transform(step_X_val)
step_X_test = xhelp.kag_pl.transform(step_X_test)

## Transform y with a label encoder fit on training labels only
step_label_encoder = preprocessing.LabelEncoder()
step_label_encoder.fit(step_y_train)
step_y_train = step_label_encoder.transform(step_y_train)
step_y_val = step_label_encoder.transform(step_y_val)
step_y_test = step_label_encoder.transform(step_y_test)
```

## Groups of Hyperparameters

```python
from sk_stepwise import Float, Int, StepwiseOptunaSearchCV

step_param_distributions = [
    {  # tree
        'max_depth': Int(1, 8),
        'min_child_weight': Float(0.1, 20, log=True),
    },
    {  # sampling
        'subsample': Float(0.5, 1.0),
        'colsample_bytree': Float(0.5, 1.0),
    },
    {  # regularization
        'reg_alpha': Float(0.0, 10.0),
        'reg_lambda': Float(0.0, 10.0),
    },
    {  # split regularization
        'gamma': Float(1e-5, 20_000, log=True),
    },
    {  # boosting
        'learning_rate': Float(1e-5, 1.0, log=True),
    },
]
```


```python
step_xg_base = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)

step_val_search = StepwiseOptunaSearchCV(
    estimator=step_xg_base,
    param_distributions=step_param_distributions,
    n_trials_per_step=30,
    random_state=42,
    verbose=1,
)
```


```python
step_val_search.fit(
    step_X_train,
    step_y_train,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)
```


```pycon
>>> print(step_val_search.best_params_)
```

```pycon
>>> # Remember to paste the parameters in so you do not
>>> # have to wait for the tuning run again.
>>> step_params = {'max_depth': 1, 'min_child_weight': 9.842315738502599, 
...   'subsample': 0.8630808426231398, 'colsample_bytree': 0.5281676324799527, 
...   'reg_alpha': 9.932142770121365, 'reg_lambda': 0.7822589021184838,
...   'gamma': 1.216698880837019e-05, 'learning_rate': 0.37469484803848585}
```


## Training a Tuned Model


```python
step_xg_val = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    **step_params,
)

step_xg_val.fit(
    step_X_train,
    step_y_train,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)
```


```pycon
>>> print(step_xg_val.score(step_X_test, step_y_test))
```


```pycon
>>> print(step_xg_val.score(step_X_train, step_y_train), 
...       step_xg_val.score(step_X_val, step_y_val))
```



```pycon
>>> step_xg_val.best_iteration
```

## Compare With a Default Model

```python
xg_val_default = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=10,
    n_estimators=500,
)

xg_val_default.fit(
    step_X_train,
    step_y_train,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)

step_scores = pd.DataFrame(
    {
        'model': ['default', 'stepwise'],
        'train': [
            xg_val_default.score(step_X_train, step_y_train),
            step_xg_val.score(step_X_train, step_y_train),
        ],
        'validation': [
            xg_val_default.score(step_X_val, step_y_val),
            step_xg_val.score(step_X_val, step_y_val),
        ],
        'test': [
            xg_val_default.score(step_X_test, step_y_test),
            step_xg_val.score(step_X_test, step_y_test),
        ],
    }
) 
```


```pycon
>>> print(step_scores)
```

## Summary
## Exercises

# Do You Have Enough Data?


## Learning Curves

```pycon
>>> # Remember to paste the parameters in so you do not
>>> # have to wait for the tuning run again.
>>> step_params = {'max_depth': 1, 'min_child_weight': 9.842315738502599, 
...   'subsample': 0.8630808426231398, 'colsample_bytree': 0.5281676324799527, 
...   'reg_alpha': 9.932142770121365, 'reg_lambda': 0.7822589021184838,
...   'gamma': 1.216698880837019e-05, 'learning_rate': 0.37469484803848585}
```

```python
import yellowbrick.model_selection as ms
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0.6, 1)
viz = ms.learning_curve(xgb.XGBClassifier(**step_params),
      step_X_train, step_y_train, ax=ax, shuffle=True, random_state=42
)

```


```python
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0.6, 1)
viz = ms.learning_curve(xgb.XGBClassifier(random_state=42),
      step_X_train, step_y_train, ax=ax, shuffle=True, random_state=42
)

```


## Learning Curves for Decision Trees

```python
# tuned tree
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0.6, 1)
viz = ms.learning_curve(tree.DecisionTreeClassifier(max_depth=7),
      step_X_train, step_y_train, ax=ax, shuffle=True, random_state=42
)

```

## Underfit Learning Curves

```python
# underfit
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0.6, 1)
viz = ms.learning_curve(tree.DecisionTreeClassifier(max_depth=1),
      step_X_train, step_y_train, ax=ax, shuffle=True, random_state=42
)

```

## Overfit Learning Curves


```python
# overfit
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0.6, 1)
viz = ms.learning_curve(tree.DecisionTreeClassifier(),
      step_X_train, step_y_train, ax=ax, shuffle=True, random_state=42
)

```


## Summary
## Exercises

# Model Evaluation
## Accuracy

```python
xgb_def = xgb.XGBClassifier()
xgb_def.fit(step_X_train, step_y_train)
```


```pycon
>>> xgb_def.score(step_X_test, step_y_test)
```


```pycon
>>> from sklearn import metrics
>>> metrics.accuracy_score(step_y_test, xgb_def.predict(step_X_test))
```


## Confusion Matrix



```pycon
>>> from sklearn import metrics
>>> cm = metrics.confusion_matrix(step_y_test, xgb_def.predict(step_X_test))
>>> cm
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=['DS', 'SE'])
ax.set_title('Confusion Matrix for Default XGBoost')
disp.plot(ax=ax, cmap='Blues')
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
cm = metrics.confusion_matrix(y_test, xgb_def.predict(X_test), 
                             normalize='true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=['DS', 'SE'])   
ax.set_title('Normalized Confusion Matrix for Default XGBoost')                    
disp.plot(ax=ax, cmap='Blues')
```


## Precision and Recall


```pycon
>>> metrics.precision_score(step_y_test, xgb_def.predict(step_X_test))
```


```pycon
>>> metrics.recall_score(step_y_test, xgb_def.predict(step_X_test))
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
disp = metrics.PrecisionRecallDisplay.from_estimator(xgb_def, 
    step_X_test, step_y_test, ax=ax)
# add train PR curve
metrics.PrecisionRecallDisplay.from_estimator(xgb_def, 
    step_X_train, step_y_train, ax=ax, label='train')
ax.set_title('Precision-Recall Curve for Default XGBoost')
```



```python
step_xg_val = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    **step_params,
)

step_xg_val.fit(
    step_X_train,
    step_y_train,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)
```

```python
fig, ax = plt.subplots(figsize=(6, 3))

disp = metrics.PrecisionRecallDisplay.from_estimator(
    step_xg_val, step_X_test, step_y_test, ax=ax)
# add train PR curve
metrics.PrecisionRecallDisplay.from_estimator(
    step_xg_val, step_X_train, step_y_train, ax=ax, label='train')
ax.set_title('Precision-Recall Curve for Tuned XGBoost')
```




## Diagnosing a Sharp Early Drop in the Precision-Recall Curve

```pycon
>>> step_probs = step_xg_val.predict_proba(step_X_train)[:, 1]
>>> step_preds = step_xg_val.predict(step_X_train)
 
>>> pr_failures = (step_X_train
...     .loc[:, ['age', 'education', 'years_exp', 'compensation', 'python',]]
...     .assign(
...         true=step_y_train,
...         pred=step_preds,
...         prob_positive=step_probs,
...         true_label=lambda df: label_encoder.inverse_transform(df.true),
...         pred_label=lambda df: label_encoder.inverse_transform(df.pred),
...     )
... )
>>> print(pr_failures
...     .query('true == 0 and pred == 1')
...     .sort_values('prob_positive', ascending=False)
...     .loc[:, ['true_label', 'pred_label', 'prob_positive', 
...    'age', 'education', 'years_exp', ]]
...     .head(10)
... )


```


```pycon
>>> print(raw
...   .loc[[16846, 16382], ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9']]   
...   .T)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))

X_data = step_X_train.loc[lambda df: ~df.index.isin([16846, 16382])]
y_data = pd.Series(step_y_train, index=step_X_train.index).loc[X_data.index]
                          
metrics.PrecisionRecallDisplay.from_estimator(
    step_xg_val, X_data, y_data,
      ax=ax)
ax.set_title('Precision-Recall Curve without False Positives')
```



```pycon
>>> print(pr_failures
...     .query('true == 1 and pred == 0')
...     .sort_values('prob_positive', ascending=True)
...     .loc[:, ['true_label', 'pred_label', 'prob_positive', 
...    'age', 'education', 'years_exp', ]]
...     .head(10)
... )
```


## F1 Score

```pycon
>>> metrics.f1_score(y_test, xgb_def.predict(X_test))
```



```pycon
>>> print(metrics.classification_report(step_y_test, 
...     y_pred=step_xg_val.predict(step_X_test), target_names=['DS', 'SE']))
```

## ROC Curve


```python
fig, ax = plt.subplots(figsize=(6, 3))
metrics.RocCurveDisplay.from_estimator(xgb_def,
                       step_X_test, step_y_test,ax=ax, label='default')
metrics.RocCurveDisplay.from_estimator(step_xg_val,
                       step_X_test, step_y_test,ax=ax)
ax.set_title('ROC Curve Comparison')
```




```python
fig, axes = plt.subplots(figsize=(6, 3), ncols=2)
metrics.RocCurveDisplay.from_estimator(xgb_def,
                       step_X_train, step_y_train,ax=axes[0], label='default train')
metrics.RocCurveDisplay.from_estimator(xgb_def,
                       step_X_test, step_y_test,ax=axes[0])
axes[0].set(title='ROC plots for default model')

metrics.RocCurveDisplay.from_estimator(step_xg_val,
                       step_X_train, step_y_train,ax=axes[1], label='step train')
metrics.RocCurveDisplay.from_estimator(step_xg_val,
                       step_X_test, step_y_test,ax=axes[1])
axes[1].set(title='ROC plots for stepwise model')
```



## Matthew’s Correlation and Cohen’s Kappa

```pycon
>>> y_pred = xgb_def.predict(step_X_test)
>>> metrics.matthews_corrcoef(step_y_test, y_pred)

```


## Cohen’s Kappa

```pycon
>>> metrics.cohen_kappa_score(step_y_test, y_pred)
```


## Threshold Metrics

```python
from sklearn.model_selection import TunedThresholdClassifierCV

xgb_def = xgb.XGBClassifier()
ttc = TunedThresholdClassifierCV(
    estimator=xgb_def,
    scoring="f1",
    cv=5,
)
ttc.fit(step_X_train, step_y_train)

```


```pycon
>>> xgb_def = xgb.XGBClassifier()
>>> xgb_def.fit(step_X_train, step_y_train)
>>> xgb_def.predict_proba(step_X_test.iloc[[0]])


```


```pycon
>>> xgb_def.predict(step_X_test.iloc[[0]])

```


```pycon
>>> ttc = TunedThresholdClassifierCV(
...     estimator=xgb.XGBClassifier(verbosity=0),
...     scoring="f1",
...     cv=5,
... )
>>> ttc.fit(step_X_train, step_y_train)
>>> ttc.best_threshold_

```


```python
# default f1 score
>>> metrics.f1_score(step_y_test, xgb_def.predict(step_X_test))
```


```python
# thresholded f1 score
>>> metrics.f1_score(step_y_test, ttc.predict(step_X_test))
```




```python
def get_tpr_fpr(preds, y_truth):
    """
    Calculate true positive rate (TPR) and false positive rate (FPR)
    given predicted labels and ground truth labels.

    Parameters
    ----------
    preds : np.array
        Predicted class labels.
    y_truth : np.array
        Ground truth labels.

    Returns
    -------
    tuple
        (tpr, fpr)
    """
    tp = (preds == 1) & (y_truth == 1)
    tn = (preds == 0) & (y_truth == 0)
    fp = (preds == 1) & (y_truth == 0)
    fn = (preds == 0) & (y_truth == 1)
    tpr = tp.sum() / (tp.sum() + fn.sum())
    fpr = fp.sum() / (fp.sum() + tn.sum())
    return tpr, fpr


vals = []
probs = ttc.estimator_.predict_proba(step_X_test)[:, 1]

for thresh in np.arange(0, 1, step=.05):
    preds = probs > thresh
    tpr, fpr = get_tpr_fpr(preds, step_y_test)
    val = [thresh, tpr, fpr]
    for metric in [
        metrics.accuracy_score,
        metrics.precision_score,
        metrics.recall_score,
        metrics.f1_score,
        metrics.roc_auc_score,
        metrics.matthews_corrcoef,
        metrics.cohen_kappa_score,
    ]:
        if metric is metrics.roc_auc_score:
            val.append(metric(step_y_test, probs))
        else:
            val.append(metric(step_y_test, preds))
    vals.append(val)

```

```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.DataFrame(
    vals,
    columns=['thresh', 'tpr/rec', 'fpr', 'acc', 'prec', 
             'rec', 'f1', 'auc', 'mcc', 'kappa']
 )
 .drop(columns='rec')
 .set_index('thresh')
 .plot(ax=ax, title='Threshold Metrics')
 .legend(bbox_to_anchor=(1, 1))
)

ax.axvline(ttc.best_threshold_, linestyle='--')
# add y-axis label
ax.set_ylabel('Score')
```


## Cumulative Gains Curve


```python
import scikitplot
fig, ax = plt.subplots(figsize=(6, 3))
y_probs = xgb_def.predict_proba(step_X_test)
scikitplot.metrics.plot_cumulative_gain(step_y_test, y_probs, ax=ax)
ax.plot([0, (step_y_test == 1).mean(), 1], [0, 1, 1], label='Optimal Class 1')
ax.set_ylim(0, 1.05)
ax.annotate('Reach 60% of\nClass 1\nby contacting top 36%', xy=(.36, .6),
           xytext=(.55,.25), arrowprops={'color':'k'})
ax.legend(bbox_to_anchor=(1, 1))
```



## Lift Curves


```python
fig, ax = plt.subplots(figsize=(6, 3))
y_probs = xgb_def.predict_proba(step_X_test)
scikitplot.metrics.plot_lift_curve(step_y_test, y_probs, ax=ax)
mean = (step_y_test == 1).mean()
ax.plot([0, mean, 1], [1/mean, 1/mean, 1], label='Optimal Class 1')
ax.legend()
```


## Looking at the Biggest Sample Failures

```pycon
>>> probs = xgb_def.predict_proba(step_X_test)[:, 1]
>>> preds = xgb_def.predict(step_X_test)

>>> print(step_X_test
...     .loc[:, ['age', 'education', 'years_exp', 'compensation', 'python']]
...     .assign(
...         true=step_y_test,
...         pred=preds,
...         prob_positive=probs,
...         correct=lambda df: df.true.eq(df.pred),
...         row_log_loss=lambda df: -(
...             df.true * np.log(np.clip(df.prob_positive, 1e-15, 1 - 1e-15))
...             + (1 - df.true) * np.log(
...                 np.clip(1 - df.prob_positive, 1e-15, 1 - 1e-15))
...         ),
...         true_label=lambda df: label_encoder.inverse_transform(df.true),
...         pred_label=lambda df: label_encoder.inverse_transform(df.pred),
...     )
...     .query('correct == False')
...     .sort_values('row_log_loss', ascending=False)
...     .loc[:, ['true_label', 'pred_label', 'prob_positive', 'row_log_loss',
...             'age', 'education', 'years_exp', 'compensation', 'python']]
...     .head(10)
... )
```


## Checking Consistency Across Cross-Validation Folds

```pycon
>>> cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
>>> cv_results = model_selection.cross_validate(
...     xgb.XGBClassifier(random_state=42),
...     step_X_train,
...     step_y_train,
...     cv=cv,
...     scoring=['accuracy', 'roc_auc', 'f1'],
...     return_train_score=True,
... )

>>> print(pd.DataFrame(cv_results)
...  .filter(regex='train_|test_')
...  .rename(columns=lambda c: c.replace('test_', '').replace('train_', 'train_'))
... )
```


```pycon
>>> print(pd.DataFrame(cv_results)
...  .filter(regex='train_|test_')
...  .agg(['mean', 'std', 'min', 'max'])
...  .T
... )

```


```pycon
>>> fold_diagnostics = []
>>> for fold, (_, test_idx) in enumerate(cv.split(step_X_train, step_y_train), start=1):
...     y_fold = pd.Series(step_y_train, index=step_X_train.index).iloc[test_idx]
...     fold_diagnostics.append({
...         'fold': fold,
...         'rows': len(test_idx),
...         'positive_rate': y_fold.mean(),
...     })

>>> print(pd.DataFrame(fold_diagnostics))
```


## Summary

## Exercises

# Training For Different Metrics
## Training with Validation Curves



```python
from yellowbrick import model_selection as ms
fig, ax = plt.subplots(figsize=(6, 3))
ms.validation_curve(xgb.XGBClassifier(), step_X_train, step_y_train,
    scoring='accuracy', param_name='learning_rate', 
    param_range=[0.001, .01, .05, .1, .2, .5, .9, 1], ax=ax
)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
ms.validation_curve(xgb.XGBClassifier(), step_X_train, step_y_train,
    scoring='roc_auc', param_name='learning_rate',
    param_range=[0.001, .01, .05, .1, .2, .5, .9, 1], ax=ax
    )
```



## Step-wise ROC Tuning

```python
from sk_stepwise import Float, Int, StepwiseOptunaSearchCV

params = {'random_state': 42}

rounds = [
    {  # tree
        'max_depth': Int(1, 9),
        'min_child_weight': Float(0.1, 20, log=True),
    },
    {  # stochastic
        'subsample': Float(0.5, 1.0),
        'colsample_bytree': Float(0.5, 1.0),
    },
    {  # regularization
        'gamma': Float(0.00004, 22_000, log=True),
    },
    {  # boosting
        'learning_rate': Float(0.0009, 1.0, log=True),
    },
]

xgb_base = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)

search = StepwiseOptunaSearchCV(
    estimator=xgb_base,
    param_distributions=rounds,
    n_trials_per_step=40,
    scoring='roc_auc',
    random_state=42,
    verbose=1,
)

search.fit(
    step_X_train,
    step_y_train,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)
```


```pycon
>>> xgb_def = xgb.XGBClassifier()
>>> xgb_def.fit(step_X_train, step_y_train)
>>> metrics.roc_auc_score(step_y_test, xgb_def.predict_proba(step_X_test)[:, 1])

```

```pycon
>>> search.best_params_
```

```pycon
>>> # the values from above training
>>> params ={'max_depth': 2,
...   'min_child_weight': 12.15184602956497,
...   'subsample': 0.9458639740440414,
...   'colsample_bytree': 0.557931022636926,
...   'gamma': 0.12289756547692056,
...   'learning_rate': 0.1760196805058194}
```


```pycon
>>> xgb_tuned = xgb.XGBClassifier(**params, early_stopping_rounds=50,
...    n_estimators=2_000)
>>> xgb_tuned.fit(step_X_train, step_y_train, eval_set=[(step_X_train, step_y_train),
...     (step_X_val, step_y_val)], verbose=100)
```


```pycon
>>> metrics.roc_auc_score(step_y_test, xgb_tuned.predict_proba(step_X_test)[:, 1])

```


## Summary
## Exercises

# Model Interpretation

## Logistic Regression Interpretation


```pycon
>>> from sklearn import linear_model, preprocessing
>>> std = preprocessing.StandardScaler()
>>> lr = linear_model.LogisticRegression(C=np.inf)
>>> lr.fit(std.fit_transform(step_X_train), step_y_train)
>>> lr.score(std.transform(step_X_val), step_y_val)
```



```pycon
>>> lr.coef_
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(lr.coef_[0], index=step_X_train.columns)
 .sort_values()
 .plot.barh(ax=ax, title='Logistic Regression Coefficients')
)
```



## Decision Tree Interpretation

```pycon
>>> tree7 = tree.DecisionTreeClassifier(max_depth=7)
>>> tree7.fit(step_X_train, step_y_train)
>>> tree7.score(step_X_val, step_y_val)
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(tree7.feature_importances_, index=step_X_train.columns)
 .sort_values()
 .plot.barh(ax=ax, title='Feature Importances from Decision Tree with max_depth=7')
)
```



```python
import dtreeviz
dt3 = tree.DecisionTreeClassifier(max_depth=3)
dt3.fit(step_X_train, step_y_train)

viz = dtreeviz.model(dt3, X_train=step_X_train, y_train=step_y_train, 
    feature_names=list(step_X_train.columns), target_name='Job',
    class_names=['DS', 'SE'])
viz.view()
```


## XGBoost Feature Importance

```python
xgb_def = xgb.XGBClassifier()
xgb_def.fit(step_X_train, step_y_train)
```

```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(xgb_def.feature_importances_, index=step_X_train.columns)
 .sort_values()
 .plot.barh(ax=ax, title='Feature Importances from XGBoost')
)
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
xgb.plot_importance(xgb_def, importance_type='cover', ax=ax)
```


## Random Features as a Sanity Check

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

X_train_rand = (
    step_X_train
    .assign(
        rand_num1=rng.normal(size=step_X_train.shape[0]),
        rand_num2=rng.normal(size=step_X_train.shape[0]),
        rand_num3=rng.uniform(size=step_X_train.shape[0]),
        rand_cat1=rng.choice(['a', 'b', 'c'], size=step_X_train.shape[0]),
        rand_cat2=rng.choice(['x', 'y', 'z', 'w'], size=step_X_train.shape[0]),
    )
    .astype({'rand_cat1': 'category', 'rand_cat2': 'category'})
)

X_test_rand = (
    step_X_test
    .assign(
        rand_num1=rng.normal(size=step_X_test.shape[0]),
        rand_num2=rng.normal(size=step_X_test.shape[0]),
        rand_num3=rng.uniform(size=step_X_test.shape[0]),
        rand_cat1=rng.choice(['a', 'b', 'c'], size=step_X_test.shape[0]),
        rand_cat2=rng.choice(['x', 'y', 'z', 'w'], size=step_X_test.shape[0]),
    )
    .astype({'rand_cat1': 'category', 'rand_cat2': 'category'})
)   

X_val_rand = (
    step_X_val
    .assign(
        rand_num1=rng.normal(size=step_X_val.shape[0]),
        rand_num2=rng.normal(size=step_X_val.shape[0]),
        rand_num3=rng.uniform(size=step_X_val.shape[0]),
        rand_cat1=rng.choice(['a', 'b', 'c'], size=step_X_val.shape[0]),
        rand_cat2=rng.choice(['x', 'y', 'z', 'w'], size=step_X_val.shape[0]),
    )
    .astype({'rand_cat1': 'category', 'rand_cat2': 'category'})
)


```



```python
xg_rand = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    enable_categorical=True,
)

xg_rand.fit(
    X_train_rand,
    step_y_train,
    eval_set=[(X_train_rand, step_y_train), (X_test_rand, step_y_test)],
    verbose=False,
)

```


```pycon
>>> fi = (
...    pd.Series(xg_rand.feature_importances_, 
...                 index=X_train_rand.columns)
...    .sort_values(ascending=False)
... )
...
>>> fi
```





```python
import sk_stepwise

from sk_stepwise import Float, Int, StepwiseOptunaSearchCV


params = {'random_state': 42}

rounds = [
    {  # tree
        'max_depth': Int(1, 9),
        'min_child_weight': Float(0.135335283, 20.085536923, log=True),
    },
    {  # stochastic
        'subsample': Float(0.5, 1.0),
        'colsample_bytree': Float(0.5, 1.0),
    },
    {  # regularization
        'gamma': Float(0.00004539993, 22026.4657948, log=True),
    },
    {  # boosting
        'learning_rate': Float(0.00091188197, 1.0, log=True),
    },
]

xgb_base = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    enable_categorical=True,
)

search = StepwiseOptunaSearchCV(
    estimator=xgb_base,
    param_distributions=rounds,
    n_trials_per_step=40,
    scoring='roc_auc',
    random_state=42,
    verbose=1,
)

search.fit(
    X_train_rand,
    step_y_train,
    eval_set=[(X_train_rand, step_y_train), (X_val_rand, step_y_val)],
    verbose=False,
)
```


```pycon
>>> search.best_params_
```

```python
params = {'max_depth': 2,
 'min_child_weight': 1.036785798266367,
 'subsample': 0.9975259140428794,
 'colsample_bytree': 0.8223368426340427,
 'gamma': 0.0010285274223717589,
 'learning_rate': 0.018840132446019068}
```


```pycon
>>> xg_step_rand = xgb.XGBClassifier(**params, random_state=42, 
...        early_stopping_rounds=50,
...        n_estimators=2_000, enable_categorical=True)
>>> xg_step_rand.fit(X_train_rand, step_y_train, eval_set=[(X_train_rand, step_y_train),
...     (X_val_rand, step_y_val)], verbose=100)

>>> fi = (pd.Series(xg_step_rand.feature_importances_, index=X_train_rand.columns)
...          .sort_values(ascending=False))
>>> fi
```


```python
subset_cols = ['r', 'major_cs', 'country_United States of America', 'major_eng',
       'education', 'years_exp', 'major_stat', 'sql', 'compensation', 'age']
subset_model = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True)
subset_model.fit(step_X_train[subset_cols], step_y_train, 
    eval_set=[(step_X_train[subset_cols], step_y_train),
              (step_X_val[subset_cols], step_y_val)], verbose=100)
```


```pycon
>>> subset_model.score(step_X_test[subset_cols], step_y_test)
```


```pycon
>>> xg_step_rand.score(X_test_rand, step_y_test)
```


```pycon
>>> metrics.roc_auc_score(step_y_test, subset_model.predict_proba(
...    step_X_test[subset_cols])[:, 1])
```


```pycon
>>> metrics.roc_auc_score(step_y_test, xg_step_rand.predict_proba(X_test_rand)[:, 1])
```


## Surrogate Models

```python
from sklearn import tree

sur_reg_sk = tree.DecisionTreeRegressor(max_depth=4)
sur_reg_sk.fit(step_X_train, xgb_def.predict_proba(step_X_train)[:,-1])
```




## Summary
## Exercises

# Sample Weights and Feature Weights
## Sample Weights

```python
params = step_params = {'max_depth': 1, 'min_child_weight': 9.842315738502599, 
   'subsample': 0.8630808426231398, 'colsample_bytree': 0.5281676324799527, 
   'reg_alpha': 9.932142770121365, 'reg_lambda': 0.7822589021184838,
   'gamma': 1.216698880837019e-05, 'learning_rate': 0.37469484803848585}


xg_step = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True)
xg_step.fit(step_X_train, step_y_train, eval_set=[(step_X_train, step_y_train),
    (step_X_val, step_y_val)], verbose=100)
abs_err = pd.Series(xg_step.predict_proba(step_X_train)[:, 1] - step_y_train, \
                    index=step_X_train.index).abs() 

```


```pycon
>>> xg_step.score(step_X_test, step_y_test)
```


```python
xg_sample_weight = xgb.XGBClassifier(**params, 
    random_state=42, early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True)

xg_sample_weight.fit(step_X_train, step_y_train, eval_set=[(step_X_train, step_y_train),
    (step_X_val, step_y_val)], verbose=100, sample_weight=abs_err)
```

```python
# score
>>> xg_sample_weight.score(step_X_test, step_y_test)
```


```python
# plot pr curve
fig, ax = plt.subplots(figsize=(6, 3))
metrics.PrecisionRecallDisplay.from_estimator(xg_sample_weight, step_X_test, 
                                              step_y_test, ax=ax)
metrics.PrecisionRecallDisplay.from_estimator(xg_sample_weight, step_X_train, 
                                              step_y_train, ax=ax, label='train')
```


```python
# plot pr curve
fig, ax = plt.subplots(figsize=(6, 3))
metrics.PrecisionRecallDisplay.from_estimator(xg_step, step_X_test, 
    step_y_test, ax=ax)
metrics.PrecisionRecallDisplay.from_estimator(xg_step, step_X_train, 
    step_y_train, ax=ax, label='train')

```



```python
import numpy as np
import xgboost as xgb
from sklearn import metrics

row_weights = np.where(step_y_train == 1, 5, 1)

xgb_w = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)

xgb_w.fit(
    step_X_train,
    step_y_train,
    sample_weight=row_weights,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    verbose=False,
)
```


```pycon
>>> xgb_def = xgb.XGBClassifier(random_state=42)
>>> xgb_def.fit(step_X_train, step_y_train)
>>> print(metrics.classification_report(step_y_test,
...            xgb_def.predict(step_X_test)))
```


```pycon
>>> xgb_w = xgb.XGBClassifier(random_state=42)
>>> xgb_w.fit(step_X_train, step_y_train, sample_weight=row_weights)
>>> print(metrics.classification_report(step_y_test, xgb_w.predict(step_X_test)))
```


```python
source_weights = np.where(X_train.source == "trusted", 1.0, 0.25)

xgb_quality = xgb.XGBClassifier(random_state=42)
xgb_quality.fit(X_train, y_train, sample_weight=source_weights)
```

## Weighted Evaluation Sets

```python
train_weights = np.where(step_y_train == 1, 5, 1)
val_weights = np.where(step_y_val == 1, 5, 1)

xgb_eval_w = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)

xgb_eval_w.fit(
    step_X_train,
    step_y_train,
    sample_weight=train_weights,
    eval_set=[(step_X_train, step_y_train), (step_X_val, step_y_val)],
    sample_weight_eval_set=[train_weights, val_weights],
    verbose=100,
)
```


## Feature Weights

```python
params = step_params = {'max_depth': 1, 'min_child_weight': 9.842315738502599, 
   'subsample': 0.8630808426231398, 'colsample_bytree': 0.5281676324799527, 
   'reg_alpha': 9.932142770121365, 'reg_lambda': 0.7822589021184838,
   'gamma': 1.216698880837019e-05, 'learning_rate': 0.37469484803848585}

feature_weights = [10 if col in ['age', 'education']
                   else 1 for col in step_X_train.columns]
xg_feat_weight = xgb.XGBClassifier(**params,
    random_state=42, early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True,
    feature_weights=feature_weights,
)
xg_feat_weight.fit(step_X_train, step_y_train, eval_set=[(step_X_train, step_y_train),
    (step_X_val, step_y_val)], verbose=100)
```

```python
xg_feat_weight.score(step_X_test, step_y_test)
```


```pycon
>>> with pd.option_context('display.max_rows', None, 'display.max_columns', None):
...     print(pd.Series(xg_feat_weight.feature_importances_, index=step_X_train.columns)
...          .sort_values(ascending=False))
```


```pycon
>>> with pd.option_context('display.max_rows', None, 'display.max_columns', None):
...     print(pd.Series(xg_step.feature_importances_, index=step_X_train.columns)
...          .sort_values(ascending=False))
```


## A Few Cautions
## Summary
## Exercises


# Feature Interactions with `treefi`
## Feature Interactions
## treefi

```python
import treefi
import importlib
importlib.reload(treefi)

xgb_rand = xgb.XGBClassifier(random_state=42, 
    early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True)
xgb_rand.fit(X_train_rand, step_y_train, 
    eval_set=[(X_train_rand, step_y_train), (X_val_rand, step_y_val)], verbose=False)  

importance = treefi.feature_importance(
    xgb_rand,
    sort_by='gain',
    top_k=50,
)

```


```pycon
>>> with pd.option_context('display.max_rows', None, 'display.max_columns', None):
...     print(importance
...         .sort_values('gain', ascending=False)
...         .round(2)
...     )
```


## `treefi` Metrics

## Pairwise Interactions

```python
interactions = treefi.feature_interactions(
    xgb_rand,
    max_interaction_depth=1,
)

```

```python
xgb_rand.
```


```pycon
>>> with pd.option_context('display.max_rows', None, 'display.max_columns', None,
...                           'display.max_colwidth', None):
...     print(interactions
...         .query('interaction_order > 1')
...         .sort_values(by='gain', ascending=False)
...         .head(10)
...         .round(1)
... )
```


```python
(X_train
 .assign(software_eng=y_train)
 .corr(method='spearman')
 .loc[:, ['education', 'years_exp', 'major_cs', 'r', 'compensation', 'age']]
 .style
 .background_gradient(cmap='RdBu', vmin=-1, vmax=1)
 .format('{:.2f}')
)

```


```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(X_train
            .assign(software_eng=y_train)
            .corr(method='spearman')
            .loc[:, ['age', 'education', 'years_exp', 'compensation', 'r',
                     'major_cs', 'software_eng']],
            cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax
)

```


```python
import seaborn.objects as so
fig = plt.figure(figsize=(6, 3))
(so
 .Plot(X_train.assign(software_eng=y_train), x='years_exp', y='education',
       color='software_eng')
 .add(so.Dots(alpha=.9, pointsize=2), so.Jitter(x=.7, y=1))
 .add(so.Line(), so.PolyFit())
 .scale(color='viridis')
 .on(fig)
 .plot()
)

```


```pycon
>>> print(X_train
...  .assign(software_eng=y_train)
...  .groupby(['software_eng', 'r', 'major_cs'])
...  .age
...  .count()
...  .unstack()
...  .unstack()
... )
```


```pycon
>>> both = (X_train
...  .assign(software_eng=y_train)
... )
>>> print(pd.crosstab(index=both.software_eng, columns=[both.major_cs, both.r]))

```


```python
fig, ax = plt.subplots(figsize=(6, 4))
grey = '#999999'
blue = '#16a2c6'
font = 'Roboto'

data = (X_train
 .assign(software_eng=y_train)
 .groupby(['software_eng', 'r', 'major_cs'])
 .age
 .count()
 .unstack()
 .unstack())

(data
 .pipe(lambda adf: adf.iloc[:,-2:].plot(color=[grey,blue], linewidth=4, ax=ax,
                                        legend=None) and adf)
 .plot(color=[grey, blue, grey, blue], ax=ax, legend=None)
)

ax.set_xticks([0, 1], ['Data Scientist', 'Software Engineer'], font=font, size=12,
              weight=600)
ax.set_yticks([])
ax.set_xlabel('')
ax.text(x=0, y=.93, s="Count Data Scientist or Software Engineer by R/CS",
        transform=fig.transFigure, ha='left', font=font, fontsize=14, weight=1000)
ax.text(x=0, y=.83, s="(Studied CS) Thick lines\n(R) Blue", transform=fig.transFigure,
        ha='left', font=font, fontsize=10, weight=300)
for side in 'left,top,right,bottom'.split(','):
    ax.spines[side].set_visible(False)
for left,txt in zip(data.iloc[0], ['Other/No R', 'Other/R', 'CS/No R', 'CS/R']):
    ax.text(x=-.02, y=left, s=f'{txt} ({left})', ha='right', va='center',
            font=font, weight=300)
for right,txt in zip(data.iloc[1], ['Other/No R', 'Other/R', 'CS/No R', 'CS/R']):
    ax.text(x=1.02, y=right, s=f'{txt} ({right})', ha='left', va='center',
            font=font, weight=300)

fig.savefig('img/slopegraph-cs-r.png', bbox_inches='tight', dpi=600)
```

## Deeper Interactions

```pycon
>>> deep_interactions = treefi.feature_interactions(
...     xgb_rand,
...     max_interaction_depth=2,
...     sort_by='gain',
...     top_k=20,
... )
>>> with pd.option_context('display.max_rows', 10, 'display.max_columns', None,
...                           'display.max_colwidth', None):
...     print(deep_interactions
...         .query('interaction_order > 2')
...         .sort_values(by='gain', ascending=False)
...     )

```

## Stability Across Folds

```pycon
>>> cv_result = treefi.cross_validated_interactions(
...     xgb.XGBClassifier(**params, random_state=42, enable_categorical=True),
...     X_train_rand,
...     step_y_train,
...     n_splits=5,
...     top_k=10,
... )


>>> print(cv_result
...   .interaction_summary          
...   .sort_values(by='mean_expected_gain', ascending=False)
...   .loc[:, ['interaction', 'mean_expected_gain', 'std_expected_gain',
...               'rare_fold_flag', 'overfit_suspect_flag',
...               'fold_count', 'fold_presence_rate']]
... )

```

## Specifying Feature Interactions

```python
constraints = [['education', 'years_exp'], ['major_cs', 'r'],
   ['compensation', 'education'], ['age', 'education'],
   ['major_cs', 'years_exp'], ['age', 'years_exp'],
   ['age', 'compensation'], ['major_stat', 'years_exp'],
]

```


```pycon
>>> def flatten(seq):
...     res = []
...     for sub in seq:
...         res.extend(sub)
...     return res
... 
... 
>>> small_cols = sorted(set(flatten(constraints)))
>>> print(small_cols)

```


```python
from sk_stepwise import Float, Int, StepwiseOptunaSearchCV

param_distributions = [
    {  # tree
        "max_depth": Int(1, 8),
        "min_child_weight": Float(0.1, 200, log=True),
    },
    {  # sampling
        "subsample": Float(0.5, 1.0),
        "colsample_bytree": Float(0.5, 1.0),
    },
    {  # regularization
        "reg_alpha": Float(0.0, 10.0),
        "reg_lambda": Float(1.0, 10.0),
    },
    {  # split regularization
        "gamma": Float(0.0001, 20_000, log=True),
    },
    {  # boosting
        "learning_rate": Float(0.001, 1.0, log=True),
    },
]

xg_constraints = xgb.XGBClassifier(
    interaction_constraints=constraints,
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    enable_categorical=True,
)

step_search = StepwiseOptunaSearchCV(
    estimator=xg_constraints,
    param_distributions=param_distributions,
    n_trials_per_step=20,
    random_state=42,
    verbose=1,
)

step_search.fit(
    step_X_train.loc[:, small_cols],
    step_y_train,
    eval_set=[(step_X_train.loc[:, small_cols], step_y_train),
              (step_X_val.loc[:, small_cols], step_y_val)],
    verbose=False,
)

```

```python
step_search.best_params_
```

```pycon
>>> constraint_params = {'max_depth': 5,
...  'min_child_weight': 5.944568820190105,
...  'subsample': 0.7159725093210578,
...  'colsample_bytree': 0.645614570099021,
...  'reg_alpha': 9.779133120688975,
...  'reg_lambda': 6.325152723370537,
...  'gamma': 0.0035914262707893653,
...  'learning_rate': 0.1570297088405538}
>>> xg_constraint = xgb.XGBClassifier(**constraint_params, 
...     interaction_constraints=constraints,
...     random_state=42, early_stopping_rounds=50, n_estimators=2_000, 
...     enable_categorical=True)
>>> xg_constraint.fit(step_X_train.loc[:, small_cols], step_y_train, 
...    eval_set=[(step_X_train.loc[:, small_cols], step_y_train),
...              (step_X_val.loc[:, small_cols], step_y_val)], verbose=100)
>>> xg_constraint.score(step_X_test.loc[:, small_cols], step_y_test)
```


```python
import xg_helpers as xhelp
xhelp.my_dot_export(xg_constraint, num_trees=0, filename='img/constrains0_xg.dot',
              title='First Constrained Tree')

```

## Summary
## Exercises

# Exploring SHAP 
## SHAP


```python
# create basic pipeline for xgboost 
from sklearn import base, compose, pipeline

def tweak_for_xgb(raw: pd.DataFrame) -> pd.DataFrame:
    # xgboost can handle categoricals, but they must be category dtype
    return (raw
        .assign(
            sex=pd.col('Q1'),
            age=pd.col('Q2').str.slice(0, 2).astype(int),
            country=pd.col('Q3').astype('category'),
            education=pd.col('Q4').astype('category'),
            major=pd.col('Q5').astype('category'),
            years_exp=(
                pd.col('Q8')
                .str.replace('+', '', regex=False)
                .str.split('-', expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                pd.col('Q9')
                .str.replace('+', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('500000', '500', regex=False)
                .str.replace(
                    'I do not wish to disclose my approximate yearly compensation',
                    '0',
                    regex=False,
                )
                .str.split('-', expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=pd.col('Q16_Part_1')=='Python',     
            r=pd.col('Q16_Part_2')=='R',
                
            sql=
                pd.col('Q16_Part_3')=='SQL',
        )
        .loc[:, 'age':'sql']
    )

class XGBPreprocessor(base.BaseEstimator, base.TransformerMixin):
    def fit(self, X, y=None):
        self.fit_ = True
        return self

    def transform(self, X):
        return tweak_for_xgb(X)

class RandomFeaturesAdder(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, n_num=3, n_cat=2, random_state=None):
        self.n_num = n_num
        self.n_cat = n_cat
        self.random_state = random_state

    def fit(self, X, y=None):
        self.fit_ = True
        return self

    def transform(self, X):
        rng = np.random.default_rng(self.random_state)
        X_new = X.copy()
        for i in range(1, self.n_num + 1):
            X_new[f'rand_num_{i}'] = rng.normal(size=X.shape[0])
        for i in range(1, self.n_cat + 1):
            X_new[f'rand_cat_{i}'] = rng.choice(['a', 'b', 'c'], size=X.shape[0])
            X_new[f'rand_cat_{i}'] = X_new[f'rand_cat_{i}'].astype('category')
        return X_new

xgb_pipe = pipeline.Pipeline([
    ('preprocessor', XGBPreprocessor()),
    ('add_random', RandomFeaturesAdder(n_num=3, n_cat=2, random_state=42)),
])

ds_se = (raw
         .query('Q6 in ["Data Scientist", "Software Engineer"]')
)


shap_X1_raw, shap_X_test_raw, shap_y1_raw, shap_y_test_raw = model_selection.train_test_split(
    ds_se, raw.loc[ds_se.index, 'Q6']=="Software Engineer", test_size=.2, random_state=42, 
    stratify=raw.loc[ds_se.index, 'Q6'])

shap_X_train, shap_X_val, shap_y_train, shap_y_val = model_selection.train_test_split(
    shap_X1_raw, shap_y1_raw, test_size=.2, random_state=42, stratify=shap_y1_raw)

shap_X_train = xgb_pipe.fit_transform(shap_X_train)
shap_X_val = xgb_pipe.transform(shap_X_val)
shap_X_test = xgb_pipe.transform(shap_X_test_raw)
```

```python
from sk_stepwise import Float, Int, StepwiseOptunaSearchCV

param_distributions = [
    {  # tree
        "max_depth": Int(1, 8),
        "min_child_weight": Float(0.1, 200, log=True),
    },
    {  # sampling
        "subsample": Float(0.5, 1.0),
        "colsample_bytree": Float(0.5, 1.0),
    },
    {  # regularization
        "reg_alpha": Float(0.0, 10.0),
        "reg_lambda": Float(1.0, 10.0),
    },
    {  # split regularization
        "gamma": Float(0.0001, 20_000, log=True),
    },
    {  # boosting
        "learning_rate": Float(0.001, 1.0, log=True),   
    },
]

xg_random = xgb.XGBClassifier(
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
    enable_categorical=True,
)

random_search = StepwiseOptunaSearchCV(
    estimator=xg_random,
    param_distributions=param_distributions,
    n_trials_per_step=20,
    random_state=42,
    verbose=1,
)

random_search.fit(
    shap_X_train,
    shap_y_train,
    eval_set=[(shap_X_train, shap_y_train), (shap_X_val, shap_y_val)],
    verbose=False,
)
```

```python
params =  {'max_depth': 4,
 'min_child_weight': 19.806033705558757,
 'subsample': 0.9162213204002109,
 'colsample_bytree': 0.6061695553391381,
 'reg_alpha': 0.5808361216819946,
 'reg_lambda': 8.795585311974417,
 'gamma': 0.0019729469790968047,
 'learning_rate': 0.015923856987613998}

xg_shap = xgb.XGBClassifier(**params, random_state=42, 
    early_stopping_rounds=50,
    n_estimators=2_000, enable_categorical=True)
xg_shap.fit(shap_X_train, shap_y_train, eval_set=[(shap_X_train, shap_y_train), (shap_X_val, shap_y_val)],
             verbose=100)    
```



```python
import shap
shap.initjs()

shap_ex = shap.TreeExplainer(xg_shap)
vals = shap_ex(shap_X_test)
```


```pycon
>>> shap_df = pd.DataFrame(vals.values, columns=shap_X_test.columns)
>>> print(shap_df)
```


```pycon
>>> print(pd.concat([shap_df.sum(axis='columns')
...                        .rename('pred') + vals.base_values,
...    pd.Series(shap_y_test, name='true')], axis='columns')
...    .assign(prob=lambda adf: (np.exp(adf.pred) / 
...                              (1 + np.exp(adf.pred))))
... )     
```


## Examining a Single Prediction

```pycon
>>> with pd.option_context('display.max_rows', None, 'display.max_colwidth', 30):
...     print(shap_X_test.iloc[0])
```


```pycon
>>> # predicts 1 - Software Engineer
>>> xg_shap.predict(shap_X_test.iloc[[0]])  
```


```pycon
>>> # ground truth
>>> shap_y_test.iloc[0]
```


```pycon
>>> # Since this is below zero, the default is Data Scientist
>>> shap_ex.expected_value
```


```pycon
>>> # < 0 therefore ... Data Scientist
>>> shap_ex.expected_value + vals.values[0].sum()
```


## Waterfall Plots

```python
fig = plt.figure(figsize=(6, 3))
shap.plots.waterfall(vals[0], show=False)
```



```python
def plot_histograms(df, columns, row=None, title='', color='shap'):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to plot histograms for.
    columns : list of str
        The names of the columns to plot histograms for.
    row : pandas.Series, optional
        A row of data to plot a vertical line for.
    title : str, optional
        The title to use for the figure.
    color : str, optional
        'shap' - color positive values red. Negative blue
        'mean' - above mean red. Below blue.
        None - black

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the histogram plots.    
    """
    red = '#ff0051'
    blue = '#008bfb'

    fig, ax = plt.subplots(figsize=(6, 3))
    hist = (df
     [columns]
     .hist(ax=ax, color='#bbb')
    )
    fig = hist[0][0].get_figure()
    if row is not None:
        name2ax = {ax.get_title():ax for ax in fig.axes}
        pos, neg = red, blue
        if color is None:
            pos, neg = 'black', 'black'
        for column in columns:
            if color == 'mean':
                mid = df[column].mean()
            else:
                mid = 0
            if row[column] > mid:
                c = pos
            else:
                c = neg
            name2ax[column].axvline(row[column], c=c)
    fig.tight_layout()
    fig.suptitle(title)
    return fig    
```


```python
features = ['education', 'r', 'major', 'age', 'years_exp', 
           'compensation']
fig = plot_histograms(shap_df, features, shap_df.iloc[0], 
                      title='SHAP values for row 0')
```



```python
num_cols = shap_X_test.select_dtypes('number').columns
num_cols
```

```python
num_cols = ['age', 'years_exp', 'compensation', 'rand_num_1',
       'rand_num_2', 'rand_num_3']
fig = plot_histograms(shap_X_test, num_cols, shap_X_test.iloc[0], 
                      title='Values for row 0', color='mean')
```



```python
fig, ax = plt.subplots(figsize=(6, 4))
(pd.Series(vals.values[0], index=shap_X_test.columns)
 .sort_values(key=np.abs)
 .plot.barh(ax=ax)
)
```


## A Force Plot

```python
# use matplotlib if having js issues
# blue - DS
# red - Software Engineer
# to save need both matplotlib=True, show=False
res = shap.plots.force(base_value=vals.base_values, 
                      shap_values=vals.values[0,:], features=shap_X_test.iloc[0], 
                      matplotlib=True, show=False
)
res.savefig('img/shap_forceplot0.png', dpi=600, bbox_inches='tight')
```


## Force Plot with Multiple Predictions

```python
# First n values
n = 100
# blue - DS
# red - Software Engineer
shap.plots.force(base_value=vals.base_values, 
               shap_values=vals.values[:n,:], features=shap_X_test.iloc[:n], 
               )
```


## Understanding Features with Dependence Plots

```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'education'], ax=ax, #color=vals, 
                   #x_jitter=0,
                   hist=True)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'education'], ax=ax, color=vals[:, 'age'],
                   x_jitter=1,
                   hist=True)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'years_exp'], ax=ax, color=vals[:, 'compensation'], 
                   x_jitter=1,
                   hist=True)
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'rand_num_1'], ax=ax, color=vals[:, 'age'], 
                   #x_jitter=0,
                   hist=True)
```



## Heatmaps and Correlations

```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(shap_X_test       
            .assign(software_eng=shap_y_test)
            .select_dtypes('number')

            .corr(method='spearman')
            .loc[:, ['age', 'years_exp', 'compensation', 'rand_num_1',
                  'rand_num_2', 'rand_num_3']],
            cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax
)
```



```python
import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(shap_df       
            .assign(software_eng=shap_y_test)
            .corr(method='spearman')
            .loc[:, ['age', 'years_exp', 'compensation', 'rand_num_1',
                 'rand_num_2', 'rand_num_3']],
            cmap='RdBu', annot=True, fmt='.2f', vmin=-1, vmax=1, ax=ax
)
```



## Beeswarm Plots of Global Behavior

```python
fig = plt.figure(figsize=(6, 4))
shap.plots.beeswarm(vals)
```



```python
from matplotlib import cm
fig = plt.figure(figsize=(6, 4))
shap.plots.beeswarm(vals, max_display=len(X_test.columns), color=cm.autumn_r)
```



## SHAP with No Interaction

```python
no_int_params = {'random_state': 42,
                 'max_depth': 1
}
xg_no_int = xgb.XGBClassifier(**no_int_params, early_stopping_rounds=50,
                              n_estimators=500, enable_categorical=True)
xg_no_int.fit(shap_X_train, shap_y_train,
       eval_set=[(shap_X_train, shap_y_train),
                 (shap_X_val, shap_y_val)
                ]
)
```


```pycon
>>> xg_no_int.score(shap_X_test, shap_y_test)
```

```python
shap_ind = shap.TreeExplainer(xg_no_int)
shap_ind_vals = shap_ind(shap_X_test)
```


```python
from matplotlib import cm
fig = plt.figure(figsize=(6, 3))
shap.plots.beeswarm(shap_ind_vals, max_display=len(shap_X_test.columns))
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'years_exp'], ax=ax, 
                   color=vals[:, 'age'], alpha=.5,
                   x_jitter=1)
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(shap_ind_vals[:, 'years_exp'], ax=ax,
                   color=shap_ind_vals[:, 'age'], alpha=.5,
                   x_jitter=1)
```



## Feature Selection with BorutaShap

```python
import boruta # my local library
params = {'max_depth': 2, 
    'min_child_weight': 3.755696977929539, 
    'subsample': 0.8071107925252813, 
    'colsample_bytree': 0.8665991722022603, 
    'reg_alpha': 0.20584494295802447, 
    'reg_lambda': 9.72918866945795, 
    'gamma': 0.1285490173378521, 
    'learning_rate': 0.2502971284865111}

boruta_model = xgb.XGBClassifier(
    **params,
    enable_categorical=True,
    random_state=42,
    n_estimators=1125,
)

feature_selector = boruta.BorutaShap(
    model=boruta_model,
    importance_measure='shap',
    classification=True,
)

feature_selector.fit(
    X=shap_X_train,
    y=shap_y_train,
    n_trials=50,
    sample=False,
    train_or_test='test',
    normalize=True,
    verbose=True,
)
```


```python
import seaborn as sns

sns.catplot(feature_selector.history_x, kind='boxen', orient='h', height=4, aspect=1.5,
            #order by mean of columns
            order=feature_selector.history_x.mean().sort_values(ascending=False).index
           ).set(title='Distribution of feature importance for both real and shadow features\n'
                 'across 50 iterations of BorutaShap')   

# get figure from seaborn catplot
fig = plt.gcf()
fig.savefig('img/boruta_history.png', dpi=600, bbox_inches='tight')
```


```pycon
>>> fi, shadow_fi = feature_selector.feature_importance(normalize=False)
>>> (pd.concat([pd.Series(fi, name='feature_importance', index=feature_selector.X.columns),
...            pd.Series(shadow_fi, name='shadow_importance', 
...                      index=[f'shadow_{c}' for c in feature_selector.X.columns])], axis='index')
...     .sort_values(ascending=False)
...     
...     )
```


```pycon
>>> print('Accepted:', feature_selector.accepted)
>>> print('Tentative:', feature_selector.tentative)
>>> print('Rejected:', feature_selector.rejected)
```


```python
selected_columns = feature_selector.accepted

xgb_selected = xgb.XGBClassifier(
    **params,
    enable_categorical=True,
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)

xgb_selected.fit(
    shap_X_train.loc[:, selected_columns],
    shap_y_train,
    eval_set=[
        (shap_X_train.loc[:, selected_columns], shap_y_train),
        (shap_X_val.loc[:, selected_columns], shap_y_val),
    ],
    verbose=100,
)
```


```pycon
>>> print('Selected-feature score:', xgb_selected.score(shap_X_test.loc[:, selected_columns], shap_y_test))
>>> print('Full-model score:', xg_shap.score(shap_X_test, shap_y_test))
```


## The Rashomon Effect

## A Practical Example

## Summary
## Exercises

# XGBoost Depths 1 and 2


```python
import xg_helpers as xhelp

## Create raw X and raw y
kag_X, kag_y = xhelp.get_rawX_y(raw, 'Q6')
    
## Split data    
kag_X_train, kag_X_test, kag_y_train, kag_y_test = \
    model_selection.train_test_split(
        kag_X, kag_y, test_size=.3, random_state=42, stratify=kag_y)    

## Transform X with pipeline
X_train = xhelp.kag_pl.fit_transform(kag_X_train)
X_test = xhelp.kag_pl.transform(kag_X_test)

## Transform y with label encoder
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(kag_y_train)
y_train = label_encoder.transform(kag_y_train)
y_test = label_encoder.transform(kag_y_test)

```

## A Depth-1 XGBoost Model

```pycon
>>> xg_def = xgb.XGBClassifier(random_state=42, enable_categorical=True)
>>> xg_def.fit(shap_X_train, shap_y_train)
>>> xg_def.score(shap_X_test, shap_y_test)

```


```pycon
>>> xg1 = xgb.XGBClassifier(max_depth=1, random_state=42, enable_categorical=True)
>>> xg1.fit(shap_X_train, shap_y_train)
>>> xg1.score(shap_X_test, shap_y_test)

```

## Feature Importance in XGB1

```pycon
>>> with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
...     print(pd.Series(xg1.feature_importances_, index=shap_X_train.columns)
...           .sort_values(ascending=False)
...          )

```


## XGB1 Feature Importance

```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(xg1.feature_importances_, index=shap_X_train.columns)
 .to_frame('xg1')
 .assign(xg_def=pd.Series(xg_def.feature_importances_, index=shap_X_train.columns))
 .sort_values('xg1', ascending=False)
 .plot.barh(title='Feature Importances from XGBoost with max_depth=1', ax=ax, legend=True)
)

```

## Similarities to GAMs



## XGBoost Depth 2

## Fitting a Depth-2 Model

```pycon
>>> xg2 = xgb.XGBClassifier(max_depth=2, enable_categorical=True,
...                             random_state=42)
>>> xg2.fit(shap_X_train, shap_y_train)
>>> xg2.score(shap_X_test, shap_y_test)

```

## What Changes at Depth 2

```python
xhelp.my_dot_export(
    xg2,
    num_trees=0,
    filename='img/xgb2_tree0.dot',
    title='First XGB2 Tree'
)

```


## Why XGB2 Feels Like Main Effects Plus Interactions

```pycon
>>> with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
...     print(pd.Series(xg2.feature_importances_, index=shap_X_train.columns)
...           .sort_values(ascending=False)
...          )
```


```python
import shap

shap.initjs()

shap_ex = shap.TreeExplainer(xg2)
vals = shap_ex(shap_X_test)
```

```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(vals[:, 'education'], ax=ax, color=vals[:, 'r'])
```


## Comparing XGB1 and XGB2

```python
feats = (pd.Series(xg_def.feature_importances_, index=shap_X_train.columns)
         .sort_values(ascending=False)
         .to_frame('default')
         .assign(xg1=pd.Series(xg1.feature_importances_, index=shap_X_train.columns),
                 xg2=pd.Series(xg2.feature_importances_, index=shap_X_train.columns)))
```

```python
fig, ax = plt.subplots(figsize=(6, 5))
(feats
 .sort_values('xg2', ascending=False)
 .plot.barh(title='Feature Importances from XGBoost with max_depth=2', ax=ax)
)
```

## Summary
## Exercises

# ICE, Partial Dependence, and Monotonic Constraints


## ICE Plots

```python
xgb_def = xgb.XGBClassifier(random_state=42)
xgb_def.fit(X_train, y_train)
xgb_def.score(X_test, y_test)
```


```python
from sklearn.inspection import PartialDependenceDisplay
fig, axes = plt.subplots(ncols=2, figsize=(6,3))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['r', 'education'],
    kind='individual', ax=axes)
axes[1].set_ylabel('')
```



```python
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['r', 'education'],
    centered=True,
    kind='individual', ax=axes)
axes[1].set_ylabel('')
```



```python
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
ax_h0 = axes[0].twinx()
ax_h0.hist(X_train.r, zorder=0)

ax_h1 = axes[1].twinx()
ax_h1.hist(X_train.education, zorder=0)

PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['r', 'education'],
    centered=True,
    ice_lines_kw={'zorder':10},
    kind='individual', ax=axes)
fig.tight_layout()
```




```python
def quantile_ice(clf, X, col, center=True, q=10, color='k', alpha=.5, legend=True,
                add_hist=False, title='', val_limit=10, ax=None):
  """
    Generate an ICE plot for a binary classifier's predicted probabilities split 
    by quantiles.

    Parameters:
    ----------
    clf : binary classifier
        A binary classifier with a `predict_proba` method.
    X : DataFrame
        Feature matrix to predict on with shape (n_samples, n_features).
    col : str
        Name of column in `X` to plot against the quantiles of predicted probabilities.
    center : bool, default=True
        Whether to center the plot on 0.5.
    q : int, default=10
        Number of quantiles to split the predicted probabilities into.
    color : str or array-like, default='k'
        Color(s) of the lines in the plot.
    alpha : float, default=0.5
        Opacity of the lines in the plot.
    legend : bool, default=True
        Whether to show the plot legend.
    add_hist : bool, default=False
        Whether to add a histogram of the `col` variable to the plot.
    title : str, default=''
        Title of the plot.
    val_limit : num, default=10
        Maximum number of values to test for col.
    ax : Matplotlib Axis, default=None
        Axis to plot on.

    Returns:
    -------
    results : DataFrame
        A DataFrame with the same columns as `X`, as well as a `prob` column with 
        the predicted probabilities of `clf` for each row in `X`, and a `group` 
        column indicating which quantile group the row belongs to.
  """                  
  probs = clf.predict_proba(X)
  df = (X
        .assign(probs=probs[:,-1],
               p_bin=lambda df_:pd.qcut(df_.probs, q=q, 
                                        labels=[f'q{n}' for n in range(1,q+1)])
               )
       )
  groups = df.groupby('p_bin')

  vals = X.loc[:,col].unique()
  if len(vals) > val_limit:
    vals = np.linspace(min(vals), max(vals), num=val_limit)
  res = []
  for name,g in groups:
    for val in vals:
      this_X = g.loc[:,X.columns].assign(**{col:val})
      q_prob = clf.predict_proba(this_X)[:,-1]
      res.append(this_X.assign(prob=q_prob, group=name))
  results = pd.concat(res, axis='index')     
  if ax is None:
    fig, ax = plt.subplots(figsize=(8,4))
  if add_hist:
    back_ax = ax.twinx()
    back_ax.hist(X[col], density=True, alpha=.2) 
  for name, g in results.groupby('group'):
    g.groupby(col).prob.mean().plot(ax=ax, label=name, color=color, alpha=alpha)
  if legend:
    ax.legend()
  if title:
    ax.set_title(title)
  return results
```


```python
fig, ax = plt.subplots(figsize=(6,3))
quantile_ice(xgb_def, X_train, 'education', q=10, legend=False, add_hist=True, ax=ax,
            title='ICE plot for Age')
```


## ICE Plots with SHAP

```python
import shap

fig, ax = plt.subplots(figsize=(6, 3))
  
shap.plots.partial_dependence(ind='education', 
    model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
    data=X_train.iloc[0:1000], ice=True, 
    npoints=(X_train.education.nunique()),
    pd_linewidth=0, show=False, ax=ax)
ax.set_title('ICE plot (from SHAP)')
```


## Partial Dependence Plots

```python
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['r', 'education'],
    kind='average', ax=axes)
fig.tight_layout()
```


```python
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['r', 'education'],
    centered=True, kind='both',
    # change color of lines
    line_kw={'color':'k', 'alpha':1},
    ice_lines_kw={'color':'r', 'alpha':.1},
    ax=axes)
fig.tight_layout()
```



```python
fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

PartialDependenceDisplay.from_estimator(xgb_def, X_train, 
    features=['years_exp', 'sex_Male'],
    centered=True, kind='both',
    line_kw={'color':'k', 'alpha':1},
    ice_lines_kw={'color':'r', 'alpha':.1},
    ax=axes)
fig.tight_layout()
```



## PDP with SHAP

```python
import shap

fig, ax = plt.subplots(figsize=(6, 3))

col = 'years_exp'  
shap.plots.partial_dependence(ind=col,
                             model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
                             data=X_train.iloc[0:1000], ice=False, 
                             npoints=(X_train[col].nunique()),
                             pd_linewidth=2, show=False, ax=ax)
ax.set_title('PDP plot (from SHAP)')
```



```python
fig, ax = plt.subplots(figsize=(6, 3))

col = 'years_exp'  
shap.plots.partial_dependence(ind=col, 
                             model=lambda rows: xgb_def.predict_proba(rows)[:,-1],
                             data=X_train.iloc[0:1000], ice=True, 
                             npoints=(X_train[col].nunique()),
                             model_expected_value=True,
                             feature_expected_value=True,
                             pd_linewidth=2, show=False, ax=ax)
ax.set_title('PDP plot (from SHAP) with ICE Plots')
```


## Monotonic Constraints

```python
fig, ax = plt.subplots(figsize=(6, 4))

(X_test
 .assign(target=y_test)
 .corr(method='spearman')
 .iloc[:-1]
 .loc[:,'target']
 .sort_values(key=np.abs)
 .plot.barh(title='Spearman Correlation with Target', ax=ax)
)
```



```pycon
>>> print(X_train
... .assign(target=y_train)
... .groupby('education')
... .mean()
... .loc[:, ['age', 'years_exp', 'target']]
... )
```


```pycon
>>> X_train.education.value_counts()
```


```pycon
>>> print(raw
... .query('Q3.isin(["United States of America", "China", "India"]) '
...        'and Q6.isin(["Data Scientist", "Software Engineer"])') 
... .query('Q4 == "Professional degree"')
... .pipe(lambda df_:pd.crosstab(index=df_.Q5, columns=df_.Q6))
... )
 
```


```pycon
>>> xgb_const = xgb.XGBClassifier(random_state=42,
...           monotone_constraints={'years_exp':1, 'education':-1})
>>> xgb_const.fit(X_train, y_train)
>>> xgb_const.score(X_test, y_test)
```



```python
small_cols = ['age', 'education', 'years_exp', 'compensation', 'python', 'r', 'sql',
              #'Q1_Male', 'Q1_Female', 'Q1_Prefer not to say',
              #'Q1_Prefer to self-describe', 
              'country_United States of America', 'country_India',
              'country_China', 'major_cs', 'major_other', 'major_eng', 'major_stat']
xgb_const2 = xgb.XGBClassifier(random_state=42,
          monotone_constraints={'years_exp':1, 'education':-1})
xgb_const2.fit(X_train[small_cols], y_train)
```


```pycon
>>> xgb_const2.score(X_test[small_cols], y_test)
```



```python
fig, ax = plt.subplots(figsize=(6,4))
(pd.Series(xgb_def.feature_importances_, index=X_train.columns)
 .sort_values()
 .plot.barh(ax=ax)
)
```


```python
fig, ax = plt.subplots(figsize=(6,4))
(pd.Series(xgb_const2.feature_importances_, index=small_cols)
 .sort_values()
 .plot.barh(ax=ax)
)
```

## Summary
## Exercises

# Calibration

## Calibrating a Model


```python
from sklearn.calibration import CalibratedClassifierCV
CalibratedClassifierCV?
```

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn import model_selection

X_prop, X_cal, y_prop, y_cal = model_selection.train_test_split(
    X_train, y_train, test_size=.2, random_state=42, stratify=y_train
)

xgb_cal_base = xgb.XGBClassifier(random_state=42)
xgb_cal_base.fit(X_prop, y_prop)

xgb_cal = CalibratedClassifierCV(
    FrozenEstimator(xgb_cal_base),
    method='sigmoid',
)
xgb_cal.fit(X_cal, y_cal)

xgb_cal_iso = CalibratedClassifierCV(
    FrozenEstimator(xgb_cal_base),
    method='isotonic',
)
xgb_cal_iso.fit(X_cal, y_cal)
```

## Calibration Curves

```python
from sklearn.calibration import CalibrationDisplay
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(6,6))
gs = GridSpec(4, 3)
axes = fig.add_subplot(gs[:2, :3])
display = CalibrationDisplay.from_estimator(xgb_def, X_test, y_test, 
                                            n_bins=10, ax=axes)
disp_cal = CalibrationDisplay.from_estimator(xgb_cal, X_test, y_test, 
                                      n_bins=10,ax=axes, name='sigmoid')
disp_cal_iso = CalibrationDisplay.from_estimator(xgb_cal_iso, X_test, y_test, 
                                      n_bins=10, ax=axes, name='isotonic')
row = 2
col = 0
ax = fig.add_subplot(gs[row, col])
ax.hist(display.y_prob, range=(0,1), bins=20)
ax.set(title='Default', xlabel='Predicted Prob')

ax2 = fig.add_subplot(gs[row, 1])
ax2.hist(disp_cal.y_prob, range=(0,1), bins=20)
ax2.set(title='Sigmoid', xlabel='Predicted Prob')

ax3 = fig.add_subplot(gs[row, 2])
ax3.hist(disp_cal_iso.y_prob, range=(0,1), bins=20)
ax3.set(title='Isotonic', xlabel='Predicted Prob')
fig.tight_layout()
```


```pycon
>>> xgb_cal.score(X_test, y_test)
```

```pycon
>>> xgb_cal_iso.score(X_test, y_test)
```

```pycon
>>> xgb_def.score(X_test, y_test)
```

## Principled Reliability Diagrams with Apple `relplot`

```python
from relplot import prepare_rel_diagram, plot_rel_diagram, smECE

probs_def = xgb_def.predict_proba(X_test)[:, 1]

with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(6, 4))
    rel_def = prepare_rel_diagram(probs_def, y_test)
    plot_rel_diagram(rel_def, fig=fig, ax=ax)
    ax.set_title('Principled Reliability Diagram (default XGBoost)')
```



## Venn-Abers Calibration

```python
from sklearn import model_selection
from venn_abers import VennAbersCalibrator

X_prop, X_cal, y_prop, y_cal = model_selection.train_test_split(
    X_train, y_train, test_size=.2, random_state=42, stratify=y_train
)

xgb_va_base = xgb.XGBClassifier(random_state=42)
xgb_va_base.fit(X_prop, y_prop)

va = VennAbersCalibrator(estimator=xgb_va_base, inductive=True, cal_size=0.2)
va.fit(X_cal, y_cal)
va_probs = va.predict_proba(X_test)[:, 1]
```


```python
# relplot for Venn-Abers calibrated model
with plt.style.context('default'):
    fig, ax = plt.subplots(figsize=(6, 4))
    rel_va = prepare_rel_diagram(va_probs, y_test)
    plot_rel_diagram(rel_va, fig=fig, ax=ax)
    ax.set_title('Principled Reliability Diagram (Venn-Abers)')
```


## `crepes` and Conformal Predictive Systems

```pycon
>>> from crepes import WrapClassifier

>>> crepes_clf = WrapClassifier(xgb.XGBClassifier(random_state=42))
>>> crepes_clf.fit(X_prop, y_prop)
>>> crepes_clf.calibrate(X_cal, y_cal)

>>> p_values = crepes_clf.predict_p(X_test)
>>> p_values[:5]
```


## Cumulative Accuracy Profile and Cumulative Calibration Profile


```python
def cumulative_accuracy_profile(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[order]
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    cap = np.cumsum(y_sorted) / y_sorted.sum()
    return pd.DataFrame({'population_frac': x, 'captured_positive_frac': cap})


def cumulative_calibration_profile(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[order]
    p_sorted = np.asarray(y_prob)[order]
    x = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    actual = np.cumsum(y_sorted) / len(y_sorted)
    predicted = np.cumsum(p_sorted) / len(y_sorted)
    return pd.DataFrame({
        'population_frac': x,
        'actual_cum_rate': actual,
        'predicted_cum_rate': predicted,
    })
```

```python
cap_def = cumulative_accuracy_profile(y_test, probs_def)
ccp_def = cumulative_calibration_profile(y_test, probs_def)

fig, axes = plt.subplots(ncols=2, figsize=(6, 4))
cap_def.plot(x='population_frac', y='captured_positive_frac', ax=axes[0])
axes[0].plot([0, 1], [0, 1], linestyle='--', color='grey')
axes[0].set_title('Cumulative Accuracy Profile')
axes[0].set_xlabel('Fraction of population reviewed')
axes[0].set_ylabel('Fraction of positives captured')

ccp_def.plot(x='population_frac', y='actual_cum_rate', ax=axes[1], label='actual')
ccp_def.plot(x='population_frac', y='predicted_cum_rate', ax=axes[1], label='predicted')
axes[1].set_title('Cumulative Calibration Profile')
axes[1].set_xlabel('Fraction of population reviewed')
axes[1].set_ylabel('Cumulative positive rate')
fig.tight_layout()
```



## Putting Calibration into a scikit-learn Pipeline

```python
from xg_helpers import kag_pl

xgb_pipe = pipeline.Pipeline(
    [
        ('prep', kag_pl),
        ('model', xgb.XGBClassifier(random_state=42)),
    ]
)

xgb_pipe_cal = CalibratedClassifierCV(
    xgb_pipe,
    method='isotonic',
    cv=5,
)

xgb_pipe_cal.fit(kag_X_train, y_train)
pipe_probs = xgb_pipe_cal.predict_proba(kag_X_test)[:, 1]

```



```python
from sklearn.frozen import FrozenEstimator

raw_X_prop, raw_X_cal, y_prop, y_cal = model_selection.train_test_split(
    kag_X_train, y_train, test_size=.2, random_state=42, stratify=y_train
)

# Fit the full preprocessing + model pipeline first
xgb_pipe.fit(raw_X_prop, y_prop)

# Freeze the already-fitted pipeline
xgb_pipe_cal_holdout = CalibratedClassifierCV(
    FrozenEstimator(xgb_pipe),
    method='sigmoid',
)

xgb_pipe_cal_holdout.fit(raw_X_cal, y_cal)
holdout_probs = xgb_pipe_cal_holdout.predict_proba(kag_X_test)[:, 1]
```


## Summary
## Exercises

# Feature Engineering

```pycon
>>> X_xg_train, X_xg_test, y_xg_train, y_xg_test = xhelp.get_cat_X_train_test(
...    raw, "Q6", add_random=False
... )

```

```pycon
>>> print(X_xg_train)
```

```pycon
>>> print(y_xg_train)
```

## Split First, Then Engineer Features

```pycon
>>> SEED = 42
>>> baseline_pipe = pipeline.Pipeline(
...     [
...         ("model", xgb.XGBClassifier(enable_categorical=True,
...                                     random_state=SEED)),
...     ]
... )

>>> model_selection.cross_val_score(baseline_pipe, X_xg_train, y_xg_train, 
...        cv=5).mean()
```

## Pipelines and Column Transformers

```python
from sklearn import set_config
set_config(transform_output="pandas")
```


```python
onehot_pre = compose.ColumnTransformer(
    transformers=[
        (
            "num",
            pipeline.Pipeline(
                [
                    ("impute", impute.SimpleImputer(strategy="median")),
                ]
            ),
            NUM_COLS,
        ),
        (
            "cat",
            preprocessing.OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CAT_COLS,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

onehot_pipe = pipeline.Pipeline(
    [
        ("pre", onehot_pre),
        ("model", xgb.XGBClassifier(enable_categorical=False, random_state=SEED)),
    ]
)

```


```pycon
>>> onehot_pipe.fit(X_xg_train, y_xg_train)
>>> onehot_pipe.score(X_xg_test, y_xg_test)
```


```pycon
>>> oh = onehot_pipe.named_steps["pre"]
>>> oh.get_feature_names_out()
```


```pycon
>>> oh.named_transformers_
```

```pycon
>>> oh.named_transformers_["num"].get_feature_names_out()
```



```pycon
>>> model_selection.cross_val_score(onehot_pipe, 
...                    X_xg_train, y_xg_train, cv=5).mean()
```

## How to Build a Pipeline

```python
numeric_pipe = pipeline.Pipeline(
    [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
    ]
)
```


```pycon
>>> numeric_pipe.fit(X_xg_train.loc[:, NUM_COLS], y_xg_train)
>>> print(numeric_pipe.transform(X_xg_train.loc[:, NUM_COLS])[:5])
```


```pycon
>>> numeric_pipe.fit(X_xg_train, y_xg_train)
```



```python
numeric_model_pipe = pipeline.Pipeline(
    [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
        ("model", xgb.XGBClassifier(random_state=SEED)),
    ]
)

```


```pycon
>>> # set max_depth to 3 in pipeline
>>> numeric_model_pipe.set_params(model__max_depth=3)
>>> numeric_model_pipe.fit(X_xg_train.loc[:, NUM_COLS], y_xg_train)
>>> model_selection.cross_val_score(numeric_model_pipe, 
...            X_xg_train.loc[:, NUM_COLS], y_xg_train, cv=5).mean()
```

## How to Build a ColumnTransformer

```python
onehot_pre = compose.ColumnTransformer(
    transformers=[
        (
            "num",
            pipeline.Pipeline(
                [
                    ("impute", impute.SimpleImputer(strategy="median")),
                ]
            ),
            NUM_COLS,
        ),
        (
            "cat",
            preprocessing.OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CAT_COLS,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

```




```pycon
>>> pd.options.display.expand_frame_repr = True    
>>> BASE_COLS = ['age', 'country', 'education', 'major', 'years_exp', 'compensation',
...        'python', 'r', 'sql']
>>> onehot_pre.fit(X_xg_train.loc[:, BASE_COLS], y_xg_train)
>>> print(onehot_pre.transform(X_xg_train.loc[:, BASE_COLS])[:5])
```


```pycon
>>> print(onehot_pre.get_feature_names_out()[:10])
```



```python
def rename_feature(transformer_name, feature_name):
    if transformer_name == "txt":
        return f"txt_{feature_name}"
    if transformer_name == "vals":
        return feature_name
    return f"{transformer_name}_{feature_name}"


named_onehot_pre = compose.ColumnTransformer(
    transformers=[
        (
            "vals",
            pipeline.Pipeline(
                [
                    ("impute", impute.SimpleImputer(strategy="median")),
                ]
            ),
            NUM_COLS,
        ),
        (
            "txt",
            preprocessing.OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CAT_COLS,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=rename_feature,
)
```


```pycon
>>> named_onehot_pre.fit(X_xg_train.loc[:, BASE_COLS], y_xg_train)
>>> named_onehot_pre.get_feature_names_out()[:10]
```

## Clean and Stable Inputs

```python
def cast_feature_dtypes(df):
    return (
        df
        .loc[:, BASE_COLS]
        .assign(
            country=pd.col("country").astype("category"),
            education=pd.col("education").astype("category"),
            major=pd.col("major").astype("category"),
            age=pd.col("age").astype("int16"),
            years_exp=pd.col("years_exp").astype("int16"),
            compensation=pd.col("compensation").astype("float32"),
            python=pd.col("python").astype("int8"),
            r=pd.col("r").astype("int8"),
            sql=pd.col("sql").astype("int8"),
        )
    )
```


```pycon
>>> cast_feature_dtypes(X_xg_train)
```


```python
def cast_feature_dtypes_pa(df):
    return (
        df
        .loc[:, BASE_COLS]
        .assign(
            country=pd.col("country").astype("category"),
            education=pd.col("education").astype("category"),
            major=pd.col("major").astype("category"),
            age=pd.col("age").astype("int16[pyarrow]"),
            years_exp=pd.col("years_exp").astype("int16[pyarrow]"),
            compensation=pd.col("compensation").astype("float32"),
            python=pd.col("python").astype("int8[pyarrow]"),
            r=pd.col("r").astype("int8[pyarrow]"),
            sql=pd.col("sql").astype("int8[pyarrow]"),
        )
    )
```

```pycon
>>> print(cast_feature_dtypes_pa(X_xg_train).age)
```



```pycon
>>> dtype_pipe = pipeline.Pipeline(
...    [
...        ("cast", preprocessing.FunctionTransformer(
...                cast_feature_dtypes_pa)),
...        ("model", make_xgb(enable_categorical=True)),
...    ]
... )

>>> model_selection.cross_val_score(dtype_pipe, X_xg_train, y_xg_train, cv=5).mean()
```


```pycon
>>> breaking_sample = pd.DataFrame(
...     {
...         "country": ["new_country"],
...         "education": ["Bachelor’s degree"],
...         "major": ["new_major"],
...         "age": [30],
...         "years_exp": [5],
...         "compensation": [100000],
...         "python": [True],
...         "r": [True],
...         "sql": [True],
...     }
... )
>>> dtype_pipe.fit(X_xg_train, y_xg_train) 
>>> dtype_pipe.predict(breaking_sample)
```


```python
class PreserveOriginalDtypes(BaseEstimator, TransformerMixin):
    def __init__(self, missing_cat_mapping=None, 
                 default_missing_value="__MISSING__"):
        super().__init__()
        self.missing_cat_mapping = missing_cat_mapping if \
            missing_cat_mapping is not None else {}
        self.default_missing_value = default_missing_value

    def fit(self, X, y=None):
        self.ordered_cols_ = X.columns.tolist()
        self.dtypes_ = {}
        self.category_maps_ = {}

        for col, dtype in X.dtypes.items():
            if isinstance(dtype, pd.CategoricalDtype):
                missing_value = self.missing_cat_mapping.get(col, 
                    self.default_missing_value)
                categories = list(dtype.categories)
                if missing_value not in categories:
                    categories = [*categories, missing_value]
                self.dtypes_[col] = pd.CategoricalDtype(
                    categories=categories,
                    ordered=dtype.ordered,
                )
                self.category_maps_[col] = set(dtype.categories)
            else:
                self.dtypes_[col] = dtype

        return self

    def transform(self, X):
        return (
            X
            .loc[:, self.ordered_cols_]
            .assign(
                **{col: (
                        pd.col(col)
                        .astype("string")
                        .where(
                            pd.col(col).isin(self.category_maps_[col]),
                            other=self.missing_cat_mapping.get(col,
                                self.default_missing_value),
                        )
                        .astype(self.dtypes_[col])
                    )
                    if col in self.category_maps_
                    else pd.col(col).astype(self.dtypes_[col])
                    for col in self.ordered_cols_
                }
            )
        )
```


```pycon
>>> dtype_pipe2 = pipeline.Pipeline(
...    [
...        ("cast", PreserveOriginalDtypes()),
...        ("model", make_xgb(enable_categorical=True)),
...    ]
... )

>>> dtype_pipe2.fit(X_xg_train, y_xg_train)
>>> dtype_pipe2.predict(breaking_sample)
```

## Categorical Features

```python
def shorten_val(ser, max_len=10):
    if isinstance(ser.dtype, (pd.api.types.CategoricalDtype, pd.StringDtype)):
        return ser.str[:max_len] + '...'
    return ser


def category_rate_table(df, col, target):
    return (
        df
        .loc[:, [col]]
        .assign(target=target,
          **{col: pd.col(col).pipe(shorten_val, max_len=10),
        })
        .groupby(col, observed=True)
        .agg(
            count=("target", "size"),
            software_engineer_rate=("target", "mean"),
        )
        .assign(
            software_engineer_rate=pd.col("software_engineer_rate").mul(100).round(1)
        )
        .sort_values("count", ascending=False)
    )
```


```pycon
>>> print(category_rate_table(X_xg_train, "major", y_xg_train))
```

## Grouping and Binning Categories

```python
EDUCATION_MAP = {
    "Master’s degree": 18,
    "Bachelor’s degree": 16,
    "Doctoral degree": 20,
    "Some college/university study without earning a bachelor’s degree": 13,
    "Professional degree": 19,
    "I prefer not to answer": None,
    "No formal education past high school": 12,
}


class GroupedCategoricalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, top_n=15):
        self.top_n = top_n

    def fit(self, X, y=None):
        self.top_countries_ = X["country"].value_counts().head(self.top_n).index
        return self

    def transform(self, X):
        return (
            X
            .loc[:, BASE_COLS]
            .assign(
                country_grouped=(
                    pd.col("country")
                    .astype("string")
                    .where(pd.col("country").isin(self.top_countries_), 
                           other="other")
                    .astype("category")
                ),
                education_years=(
                    pd.col("education")
                    .astype("string")
                    .map(EDUCATION_MAP)
                    .astype("int8[pyarrow]")
                ),
                education_bin=lambda df_: pd.cut(
                    df_["education_years"],
                    bins=[0, 12, 16, 18, 25],
                    labels=["hs_or_less", "college", "masters", "doctorate"],
                    include_lowest=True,
                ),
                major_family=(
                    pd.col("major")
                    .astype("string")
                    .str.split(" ")
                    .str[0]
                    .str.lower()
                    .astype("category")
                ),
            )
            .astype({"education_bin": "category"})
        )


grouped_cat_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", GroupedCategoricalFeatures(top_n=15)),
        ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
    ]
)

```


```pycon
>>> model_selection.cross_val_score(grouped_cat_pipe, X_xg_train, y_xg_train, 
...        cv=5).mean()
```


```python
class NoFitTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
```


```pycon
>>> from sklearn.utils.validation import check_is_fitted

>>> nft_pipe = pipeline.Pipeline(
...    [
...         ("nft", NoFitTransformer()),
...         ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
...     ]
... )
>>> nft_pipe.fit(X_xg_train, y_xg_train)
>>> check_is_fitted(nft_pipe.named_steps["nft"])
```


```pycon
>>> grouped_cat_pipe.fit(X_xg_train, y_xg_train)
>>> grouped_cat_pipe.predict(X_xg_test)
```


```pycon
>>> breaking_sample = pd.DataFrame(
...     {
...         "country": ["new_country"],
...         "education": ["Bachelor’s degree"],
...         "major": ["new_major"],
...         "age": [30],
...         "years_exp": [5],
...         "compensation": [100000],
...         "python": [True],
...         "r": [True],
...         "sql": [True],
...     }
... )
>>> grouped_cat_pipe.predict(breaking_sample)
```



```python
class PreserveCategories(BaseEstimator, TransformerMixin):
    def __init__(self, cols, top_n=15):
        self.cols = cols
        self.top_n = top_n

    def fit(self, X, y=None):
        self.top_categories_ = {col: X[col].value_counts().head(self.top_n).index 
                                for col in self.cols}
        return self

    def transform(self, X):
        return (
            X
            .assign(
                **{
                    col: X[col]
                    .astype("string")
                    .where(X[col].isin(self.top_categories_[col]), other="other")
                    .astype("category")
                    for col in self.cols
                }
            )
        )
    
preserve_country_pipe = pipeline.Pipeline(
    [
        ("preserve_country", PreserveCategories(cols=["country"], top_n=15)),
    ])

preserve_country_pipe.fit(X_xg_train, y_xg_train)
```



```pycon
>>> print(preserve_country_pipe.transform(breaking_sample))
```


```pycon
>>> grouped_cat_pipe = pipeline.Pipeline(
...     [
...         ("cast", PreserveOriginalDtypes()),
...         ("preserve_country", PreserveCategories(
...                cols=["country", "education", "major"], top_n=15)),
...         ("features", GroupedCategoricalFeatures(top_n=15)),
...         ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
...     ]
... )

>>> grouped_cat_pipe.fit(X_xg_train, y_xg_train)
>>> grouped_cat_pipe.predict(breaking_sample)
```

## Numeric Transformations and Bins

```python
class NumericFeatureBlock(BaseEstimator, TransformerMixin):
    def __init__(self, upper_quantile=0.99):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.comp_upper_ = X["compensation"].quantile(self.upper_quantile)
        return self

    def transform(self, X):
        return (
            X
            .loc[:, BASE_COLS]
            .assign(
                compensation_capped=(
                    pd.col("compensation").clip(upper=self.comp_upper_)
                ),
                compensation_log=np.log1p(pd.col("compensation_capped")),
                years_exp_sqrt=np.sqrt(pd.col("years_exp").clip(lower=0)),
                age_times_exp=pd.col("age") * pd.col("years_exp"),
                exp_per_age=(
                    pd.col("years_exp")
                    / pd.col("age").replace(0, np.nan)
                ),
                age_bin=lambda df_: pd.cut(
                    df_.age,
                    bins=[0, 25, 35, 45, 100],
                    labels=["18-25", "26-35", "36-45", "46+"],
                    include_lowest=True,
                ),
                exp_bin=lambda df_: pd.cut(
                    df_.years_exp,
                    bins=[-1, 1, 5, 10, 100],
                    labels=["0-1", "2-5", "6-10", "10+"],
                    include_lowest=True,
                ),
            )
        )


numeric_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", NumericFeatureBlock(upper_quantile=0.99)),
        ("model", xgb.XGBClassifier(random_state=SEED, enable_categorical=True)),
    ]
)

```

## Time and Cyclic Features

```python
def add_cyclic_time_features(df, month_col="month", period=12):
    return (
        df
        .loc[:, [month_col]]
        .assign(
            month_sin=np.sin(2 * np.pi * pd.col(month_col) / period),
            month_cos=np.cos(2 * np.pi * pd.col(month_col) / period),
        )
    )


calendar_df = pd.DataFrame({"month": [1, 2, 3, 4, 10, 11, 12]})

cyclic_pipe = pipeline.Pipeline(
    [
        (
            "features",
            preprocessing.FunctionTransformer(
                add_cyclic_time_features,
                kw_args={"month_col": "month"},
                validate=False,
            ),
        ),
    ]
)
```


```pycon
>>> print(cyclic_pipe.fit_transform(calendar_df))
```

## Row-Wise Summaries, Formulas, and Missingness

```python
def skill_block_numeric(df, skill_cols=SKILL_COLS):
    return df.loc[:, skill_cols].fillna(0).astype("float32")


def add_summary_features(df, skill_cols=SKILL_COLS):
    return (
        df
        .loc[:, BASE_COLS]
        .assign(
            tool_count=lambda df_: 
                skill_block_numeric(df_, skill_cols).sum(axis="columns"),
            tool_share=lambda df_: 
                skill_block_numeric(df_, skill_cols).mean(axis="columns"),
            tool_range=lambda df_: (
                skill_block_numeric(df_, skill_cols).max(axis="columns")
                - skill_block_numeric(df_, skill_cols).min(axis="columns")
            ),
            coding_intensity=(
                2 * pd.col("python")
                + 1.5 * pd.col("sql")
                + 1.0 * pd.col("r")
                + 0.1 * pd.col("years_exp")
            ),
            compensation_per_year=(
                pd.col("compensation")
                / (pd.col("years_exp") + 1)
            ),
            missing_age=pd.col("age").isna()
        )
    )


summary_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", preprocessing.FunctionTransformer(
            add_summary_features, validate=False)),
        ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
    ]
)
```

## Learned Encodings for Categories

```python
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = ["country", "major"] if columns is None else columns

    def fit(self, X, y=None):
        self.mappings_ = {
            col: X[col].value_counts(normalize=True)
            for col in self.columns
        }
        return self

    def transform(self, X):
        out = X
        for col, mapping in self.mappings_.items():
            out = out.assign(
                **{f"{col}_freq": pd.col(col)
                   .cat.add_categories(0.0)
                   .map(mapping).fillna(0.0)}
            )
        return out.assign(
            major_inverse_freq=(1 / pd.col("major_freq").replace(0, np.nan)).fillna(0.0)
        )
```

```python
frequency_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", FrequencyEncoder(columns=["country", "major"])),
        ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
    ]
)
```



```python
class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column="major"):
        self.column = column

    def fit(self, X, y):
        y_ser = pd.Series(y, index=X.index)
        self.mapping_ = (
            pd.DataFrame({self.column: X[self.column], "target": y_ser})
            .groupby(self.column, observed=False)["target"]
            .mean()
        )
        self.global_mean_ = y_ser.mean()
        return self

    def transform(self, X):
        return (
            X
            .loc[:, BASE_COLS]
            .assign(
                **{
                    f"{self.column}_te": (
                        pd.col(self.column)
                        .astype("string")
                        .map(self.mapping_)
                        .astype("float64")
                        .fillna(self.global_mean_)
                    )
                }

            )
            # make sure string columns are categories
                .assign(
                    **{
                        col: pd.col(col).astype("category")
                        for col in CAT_COLS
                    }
                )
        )
```


```pycon
>>> mte = MeanTargetEncoder(column="major")
>>> print(mte.fit_transform(X_xg_train, y_xg_train))
```


```python
target_encode_pipe = pipeline.Pipeline(    
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", MeanTargetEncoder(column="major")),
        ("model", xgb.XGBClassifier(enable_categorical=True, random_state=SEED)),
    ]
)

```

```python
target_encode_pipe.fit(X_xg_train, y_xg_train)  
target_encode_pipe.predict(breaking_sample)
```

## Dimensionality Reduction 

```python
pca_pipe = pipeline.Pipeline(
    [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
        ("pca", decomposition.PCA(n_components=2, random_state=SEED)),
    ]
)
```

```pycon
>>> pca_pipe.fit(X_xg_train.loc[:, NUM_COLS], y_xg_train)
>>> print(pca_pipe.transform(X_xg_train.loc[:, NUM_COLS])[:5])
```


```python
# append PCA components to original features
append_pca_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", compose.ColumnTransformer(
            transformers=[
                (
                    "num",
                    pipeline.Pipeline(
                        [
                            ("impute", impute.SimpleImputer(strategy="median")),
                            ("scale", preprocessing.StandardScaler()),
                            ("pca", decomposition.PCA(n_components=2, 
                                                      random_state=SEED)),
                        ]
                    ),
                    NUM_COLS,
                ),
                (
                    "passthrough",
                    "passthrough",
                    BASE_COLS
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        ))
    ]
)
```

```pycon
>>> append_pca_pipe.fit(X_xg_train, y_xg_train)
>>> print(append_pca_pipe.transform(X_xg_train))
```


```python
tsne_pre = pipeline.Pipeline(
    [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
        ("tsne", manifold.TSNE(
            n_components=2,
            learning_rate="auto",
            init="pca",
            random_state=SEED,
        )),
    ]
)
```

```pycon
>>> tsne_pre.fit(X_xg_train.loc[:, NUM_COLS], y_xg_train)
>>> print(tsne_pre.transform(X_xg_train.loc[:, NUM_COLS])[:5])
```


```python
import umap
umap_pre = pipeline.Pipeline(
    [
        ("impute", impute.SimpleImputer(strategy="median")),
        ("scale", preprocessing.StandardScaler()),
        ("umap", umap.UMAP(
            n_components=2,
            random_state=SEED,
        )),
    ]

)
```

```pycon
>>> umap_pre.fit(X_xg_train.loc[:, NUM_COLS], y_xg_train)
>>> print(umap_pre.transform(X_xg_train.loc[:, NUM_COLS])[:5])
```


## Domain Features and Existing Indicators

```python
def add_technical_indicators(df, close_col="close", high_col="high", low_col="low", window=5):
    return (
        df
        .loc[:, [close_col, high_col, low_col]]
        .astype({close_col: "float64", high_col: "float64", low_col: "float64"})
        .assign(
            delta=pd.col(close_col).diff(),
            gain=pd.col("delta").clip(lower=0),
            loss=(-pd.col("delta")).clip(lower=0),
            avg_gain=pd.col("gain").rolling(window=window, 
                                            min_periods=window).mean(),
            avg_loss=pd.col("loss").rolling(window=window, 
                                            min_periods=window).mean(),
            sma_5=pd.col(close_col).rolling(window=window, 
                                            min_periods=window).mean(),
            ema_5=pd.col(close_col).ewm(span=window, adjust=False).mean(),
            rolling_std=pd.col(close_col).rolling(window=window, 
                                                  min_periods=window).std(),
            rs=pd.col("avg_gain") / pd.col("avg_loss").replace(0, np.nan),
            rsi_5=100 - (100 / (1 + pd.col("rs"))),
            bollinger_upper=pd.col("sma_5") + 2 * pd.col("rolling_std"),
            bollinger_lower=pd.col("sma_5") - 2 * pd.col("rolling_std"),
            price_range=pd.col(high_col) - pd.col(low_col),
        )
        .loc[:, [close_col, high_col, low_col, "sma_5", "ema_5", "rsi_5",
                "bollinger_upper", "bollinger_lower", "price_range"]]
    )


price_df = pd.DataFrame(
    {
        "close": [100, 102, 101, 105, 107, 106, 108, 111],
        "high": [101, 103, 102, 106, 108, 107, 109, 112],
        "low": [99, 100, 99, 103, 105, 104, 106, 109],
    }
)

indicator_pipe = pipeline.Pipeline(
    [
        ("cast", PreserveOriginalDtypes()),
        ("features", preprocessing.FunctionTransformer(add_technical_indicators, 
                                                       validate=False)),
    ]
)
```



```pycon
>>> print(indicator_pipe.fit_transform(price_df))
```

## Scoring Feature Attempts

```python
def score_feature_block(name, feature_step):
    pipe = pipeline.Pipeline(
        [
            ("features", feature_step),
            ("model", make_xgb(enable_categorical=True)),
        ]
    )
    return {"feature_block": name, 
            "cv_auc": model_selection.cross_val_score(pipe, X_xg_train, 
                y_xg_train, cv=5, scoring="roc_auc").mean()}


comparison = pd.DataFrame(
    [
        score_feature_block("baseline", "passthrough"),
        score_feature_block(
            "grouped_categories",
            GroupedCategoricalFeatures(top_n=15),
        ),
        score_feature_block(
            "numeric_features",
            NumericFeatureBlock(upper_quantile=0.99),
        ),
        score_feature_block(
            "summary_features",
            preprocessing.FunctionTransformer(add_summary_features, 
                                              validate=False),
        ),
        score_feature_block(
            "frequency_encoding",
            FrequencyEncoder(columns=["country", "major"]),
        ),
    ]
)
```

```pycon
>>> print(comparison.sort_values("cv_auc", ascending=False))
```

## A Practical Workflow
## Summary
## Exercises






# Other Tabular Competitors
## A Useful Way to Compare Them
## HistGradientBoosting


```pycon
>>> from sklearn.ensemble import HistGradientBoostingClassifier
>>> hgb = HistGradientBoostingClassifier(random_state=42, 
...                categorical_features=kag_X_test.columns)
>>> hgb.fit(kag_X_train, kag_y_train)
>>> hgb.score(kag_X_test, kag_y_test)

```




```python
from xg_helpers.hist_gbm_tree_plot import plot_hist_gbm_tree

fig, ax = plot_hist_gbm_tree(
    hgb,
    feature_names=list(kag_X_test.columns),
    max_depth=4,
    output_path="histgbm_tree.png",
    figsize=(12, 10)
)

```


## LightGBM

```pycon
>>> import lightgbm as lgb

>>> cat_cols = list(kag_X_train.columns)
>>> lgb_X_train = kag_X_train.astype({col: "category" for col in cat_cols})
>>> lgb_X_test = kag_X_test.astype({col: "category" for col in cat_cols})

>>> lgbm = lgb.LGBMClassifier(random_state=42)
>>> lgbm.fit(lgb_X_train, kag_y_train)#, categorical_feature=cat_cols)
>>> lgbm.score(lgb_X_test, kag_y_test)

```


```python
#visiualization of the first tree in the LightGBM model
fig, ax = plt.subplots(figsize=(20, 10))
lgb.plot_tree(lgbm, tree_index=0, ax=ax, 
              show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
ax.set_title('LightGBM Tree Visualization')
fig.savefig('img/lgbm_tree.png', bbox_inches='tight', dpi=300)
```


## CatBoost


```pycon
>>> from catboost import CatBoostClassifier
>>> cat_cols = list(kag_X_train.columns)

>>> cat = CatBoostClassifier(
...     verbose=0,
...     random_state=42,
...     cat_features=cat_cols,
... )
>>> cat.fit(kag_X_train.fillna('missing'), kag_y_train)
>>> cat.score(kag_X_test.fillna('missing'), kag_y_test)

```


```python
# visualization of the first tree in the CatBoost model
from catboost import Pool
pool = Pool(kag_X_train.fillna('missing'), kag_y_train, cat_features=cat_cols)
digraph = cat.plot_tree(pool=pool, tree_idx=0)
# save graphviz object to PNG
digraph.save('img/catboost_tree.dot')
import graphviz
with open('img/catboost_tree.dot') as f:
    dot_graph = f.read()
graph = graphviz.Source(dot_graph)
graph.render('img/catboost_tree', format='png', cleanup=True)   
```


## TabPFN, AutoGluon, and Other Competitors


## TabICL

```pycon
>>> from sklearn import metrics
>>> from tabicl import TabICLClassifier
>>> tabicl = TabICLClassifier(
...     random_state=42,
...     verbose=False,
... )

>>> tabicl.fit(kag_X_train, kag_y_train=='Software Engineer')
```

## What About Random Forest?

```pycon
>>> from xgboost import XGBRFClassifier
>>> xg_rf = XGBRFClassifier(random_state=42)
>>> xg_rf.fit(X_train, y_train)
>>> xg_rf.score(X_test, y_test)
```


## Logistic Regression

```pycon
>>> from sklearn.linear_model import LogisticRegression
>>> from xg_helpers import kag_pl
>>> # make pipeline for logistic regression w/ standard scaling
>>> from sklearn import pipeline, preprocessing
>>> lr_pipe = pipeline.Pipeline([
...     ('preprocessor', kag_pl),
...     ('std', preprocessing.StandardScaler()),
...     ('classifier', LogisticRegression(random_state=42))
... ])
>>> lr_pipe.fit(kag_X_train, kag_y_train)
>>> lr_pipe.score(kag_X_test, kag_y_test)
```

## Summary
## Exercises

# Deploying Pipeline Models with Flask and FastAPI


## Training a Pipeline with the Kaggle Data


```python
from sklearn import base

class TweakKagTransformer(base.BaseEstimator,
    base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing 
    Kaggle survey data as input and returns a new version of 
    the DataFrame. The modifications include extracting and 
    transforming certain columns, renaming columns, and 
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable. 
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    
    def __init__(self, ycol=None):
        self.ycol = ycol
        
    def transform(self, X):
        return tweak_kag(X)
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
        
def topn(ser, n=5, default='other'):
    """
    Replace all values in a Pandas Series that are not among 
    the top `n` most frequent values with a default value.

    This function takes a Pandas Series and returns a new 
    Series with the values replaced as described above. The 
    top `n` most frequent values are determined using the 
    `value_counts` method of the input Series.

    Parameters
    ----------
    ser : pd.Series
        The input Series.
    n : int, optional
        The number of most frequent values to keep. The 
        default value is 5.
    default : str, optional
        The default value to use for values that are not among 
        the top `n` most frequent values. The default value is 
        'other'.

    Returns
    -------
    pd.Series
        The modified Series with the values replaced.
    """    
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


def get_education(ser):
    return (ser
            .replace({'Master’s degree': 18,
                         'Bachelor’s degree': 16,
                         'Doctoral degree': 20,
'Some college/university study without earning a bachelor’s degree': 13,
                         'Professional degree': 19,
                         'I prefer not to answer': None,
                         'No formal education past high school': 12})
                         .astype(float)
    )

def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Tweak the Kaggle survey data and return a new DataFrame.

    This function takes a Pandas DataFrame containing Kaggle 
    survey data as input and returns a new DataFrame. The 
    modifications include extracting and transforming certain 
    columns, renaming columns, and selecting a subset of columns.

    Parameters
    ----------
    df_ : pd.DataFrame
        The input DataFrame containing Kaggle survey data.

    Returns
    -------
    pd.DataFrame
        The new DataFrame with the modified and selected columns.
    """    
    major_map = {
        'Computer science (software engineering, etc.)': 'cs',
        'Engineering (non-computer focused)': 'eng',
        'Mathematics or statistics': 'stat',
    }

    return (
        df_
        .assign(
            sex=pd.col('Q1'),
            age=pd.col('Q2').str.slice(0, 2).astype(int),
            country=pd.col('Q3'),
            education=pd.col('Q4').pipe(get_education),
            major=pd.col('Q5').replace(major_map)
                            .where(lambda s: s.isin(major_map.values()), 'other'),
            years_exp=(
                pd.col('Q8')
                .str.replace('+', '', regex=False)
                .str.split('-', expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                pd.col('Q9')
                .str.replace('+', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace('500000', '500', regex=False)
                .str.replace(
                    'I do not wish to disclose my approximate yearly compensation',
                    '0',
                    regex=False,
                )
                .str.split('-', expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=(
                pd.col('Q16_Part_1')
                .fillna(0)
                .replace('Python', 1)
                .astype(int)
            ),
            r=(
                pd.col('Q16_Part_2')
                .fillna(0)
                .replace('R', 1)
                .astype(int)
            ),
            sql=(
                pd.col('Q16_Part_3')
                .fillna(0)
                .replace('SQL', 1)
                .astype(int)
            ),
        )
        .loc[
            :,
            ['sex', 'age', 'country', 'education', 'major', 'years_exp',
             'compensation', 'python', 'r', 'sql']
        ]
    )
```


```python
def get_rawX_y(df, y_col):
    raw = (df
            .query('Q3.isin(["United States of America", "China", "India"]) '
               'and Q6.isin(["Data Scientist", "Software Engineer"])')
            .loc[:, ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q8', 'Q9', 'Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3']]
          )
    return raw.drop(columns=[y_col]), raw[y_col]

X_raw, y_raw = get_rawX_y(raw, 'Q6')
```


```python
# label encode the target variable
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(y_raw)
y = label_encoder.transform(y_raw)
```


```python
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X_raw, y, test_size=.3, random_state=42, 
                                     stratify=y)
```


```python
params = {'max_depth': 3,
  'min_child_weight': 0.9629513813339752,
  'subsample': 0.8938412919328188,
  'colsample_bytree': 0.9729048599868567,
  'reg_alpha': 5.475582410704039,
  'reg_lambda': 4.416438190782011,
  'gamma': 0.0010285274223717589,
  'learning_rate': 0.3062671318355995}
 
```


```python
## Create a pipeline
kag_pl = pipeline.Pipeline([('tweak', TweakKagTransformer()),
    ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
           variables=['sex', 'country', 'major'])),
    ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
          variables=['education', 'years_exp'])),
    ('xgb', xgb.XGBClassifier(**params, use_label_encoder=False, 
                              eval_metric='logloss'))
])

```


```python
kag_pl.fit(X_train, y_train)
```



```pycon
>>> kag_pl.score(X_test, y_test)
```


## Serializing the Pipeline


```python
from xg_helpers import TweakKagTransformer
## Create a pipeline
kag_pl = pipeline.Pipeline([('tweak', TweakKagTransformer()),
    ('cat', encoding.OneHotEncoder(top_categories=5, drop_last=True, 
           variables=['sex', 'country', 'major'])),
    ('num_impute', imputation.MeanMedianImputer(imputation_method='median',
          variables=['education', 'years_exp'])),
('xgb', xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss'))
])

kag_pl.fit(X_train, y_train)
```


```python
import joblib
joblib.dump(kag_pl, '../build/kaggle_pipeline.joblib')
```


```pycon
>>> import joblib
>>> pipe = joblib.load('../build/kaggle_pipeline.joblib')
>>> pipe.score(X_test, y_test)
```



## What the Input Should Look Like

```python
sample = {
    "Q1": "Male",
    "Q2": "18-21",
    "Q3": "India",
    "Q4": "Bachelor’s degree",
    "Q5": "Computer science (software engineering, etc.)",
    "Q8": "1-2 years",
    "Q9": "I do not wish to disclose my approximate yearly compensation",
    "Q16_Part_1": 1,
    "Q16_Part_2": 0,
    "Q16_Part_3": 0,
}

```



```pycon
>>> pipe.predict_proba(pd.DataFrame.from_dict([sample]))
```


## A Flask App

```python

from flask import Flask, jsonify, request
import joblib
import pandas as pd

from xg_helpers import TweakKagTransformer

app = Flask(__name__)

pipe = joblib.load('kaggle_pipeline.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    rows = payload if isinstance(payload, list) else [payload]
    X_new = pd.DataFrame(rows)

    pred = pipe.predict(X_new)
    prob = pipe.predict_proba(X_new)

    return jsonify({
        'prediction': pred.tolist(),
        'probability': prob.tolist(),
    })

def main():
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)



```



## A FastAPI App




```python
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from xg_helpers import TweakKagTransformer

app = FastAPI()

MODEL_PATH = Path(__file__).parent / 'kaggle_pipeline.joblib'
pipe = joblib.load(MODEL_PATH)

class KaggleRow(BaseModel):
    Q1: str
    Q2: str
    Q3: str
    Q4: str
    Q5: str
    Q8: str
    Q9: str
    Q16_Part_1: int
    Q16_Part_2: int
    Q16_Part_3: int

@app.post('/predict')
def predict(row: KaggleRow):
    X_new = pd.DataFrame([row.model_dump()])
    pred = pipe.predict(X_new)
    prob = pipe.predict_proba(X_new)
    return {
        'prediction': pred[0],
        'probability': prob[0].tolist(),
    }

def main():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

```


## Returning Labels Instead of Encoded Classes

## Saving a Real Deployment Bundle

## Handling Errors and Missing Columns
## Docker
## A Small Batch Endpoint
## Summary
## Exercises

# Monitoring and Drift
## What People Mean by Drift
## A Simple Example

## Detecting Data Drift

```pycon
>>> train_stats = X_train.describe().round(2).T
>>> test_stats = X_test.describe().round(2).T
>>> print(train_stats-test_stats)
```



```python
import numpy as np
import pandas as pd

def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    breaks = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    expected_bins = pd.cut(expected, breaks, include_lowest=True)
    actual_bins = pd.cut(actual, breaks, include_lowest=True)

    e = expected_bins.value_counts(normalize=True).sort_index()
    a = actual_bins.value_counts(normalize=True).sort_index()

    eps = 1e-6
    return (((a + eps) - (e + eps)) * np.log((a + eps) / (e + eps))).sum()
```


```pycon
>>> psis = {}
>>> for col in X_train.select_dtypes(include=np.number).columns:
...     psis[col] = psi(X_train[col], X_test[col], bins=20)
>>> print(pd.Series(psis).sort_values(ascending=False))
```



## Data Drift with Kolmogorov-Smirnov Tests

```pycon
>>> # run ks_2samp test on each numeric column
>>> from scipy.stats import ks_2samp
>>> ks_results = {}
>>> for col in X_train.select_dtypes(include=np.number).columns:
...     ks, p_value = ks_2samp(X_train[col], X_test[col])
...     ks_results[col] = {'statistic': ks, 'p_value': p_value}
>>> print(pd.DataFrame(ks_results).T.sort_values(by='statistic', ascending=False))
```


## Data Drift with Jensen-Shannon Divergence

```pycon
>>> # run jensen-shannon divergence on each numeric column
>>> from scipy.spatial.distance import jensenshannon
>>> js_results = {}
>>> for col in X_train.select_dtypes(include=np.number).columns:
...     js = jensenshannon(X_train[col].value_counts(normalize=True),
...                         X_test[col].value_counts(normalize=True))
...     js_results[col] = js
>>> print(pd.Series(js_results).sort_values(ascending=False))   
```


## Data Drift with Wasserstein Distance

```pycon
>>> # run wasserstein_distance
>>> from scipy.stats import wasserstein_distance
>>> wd_results = {}
>>> for col in X_train.select_dtypes(include=np.number).columns:
...     wd = wasserstein_distance(X_train[col], X_test[col])
...     wd_results[col] = wd
>>> print(pd.Series(wd_results).sort_values(ascending=False))
```


## Detecting Categorical Drift

## Categorical Data Drift with the Chi-Square Test



```pycon
>>> print(pd.DataFrame({
...     "train": kag_X_train.Q9.value_counts(),
...     "test": kag_X_test.Q9.value_counts()})
... .T)
```


```pycon
>>> from scipy.stats import chi2_contingency
>>> chi2_results = {}
>>> categorical_cols = kag_X_train.select_dtypes(exclude=np.number).columns
>>> for col in categorical_cols:
...     # create contingency table
...     table = pd.DataFrame({
...         "train": kag_X_train[col].value_counts(),
...         "test": kag_X_test[col].value_counts()
...     }).T
...     statistic, p_value, dof, expected = chi2_contingency(table)
...     chi2_results[col] = {
...         "statistic": statistic,
...         "p_value": p_value,
...         "dof": dof,
...         "expected": expected
...     }
>>> print(pd.DataFrame(chi2_results)
...          .T
...          .sort_values("statistic", ascending=False))
```


## Data Drift with Jensen-Shannon Divergence

```pycon
>>> from scipy.spatial.distance import jensenshannon
>>> js_results = {}
>>> categorical_cols = kag_X_train.select_dtypes(exclude=np.number).columns
>>> for col in categorical_cols:
...     train_counts = kag_X_train[col].value_counts(normalize=True)
...     test_counts = kag_X_test[col].value_counts(normalize=True)
...
...     categories = train_counts.index.union(test_counts.index)
...     train_probs = train_counts.reindex(categories, fill_value=0)
...     test_probs = test_counts.reindex(categories, fill_value=0)
...
...     js_distance = jensenshannon(train_probs, test_probs)
...     js_results[col] = js_distance ** 2
>>> print(pd.Series(js_results, name='Jensen-Shannon Divergence Squared')
...          .sort_values(ascending=False))
```


## Detecting Prediction Drift



```python
params = {'max_depth': 3,
  'min_child_weight': 0.9629513813339752,
  'subsample': 0.8938412919328188,
  'colsample_bytree': 0.9729048599868567,
  'reg_alpha': 5.475582410704039,
  'reg_lambda': 4.416438190782011,
  'gamma': 0.0010285274223717589,
  'learning_rate': 0.3062671318355995}
 

xg_step = xgb.XGBClassifier(
    **params,
    random_state=42,
    early_stopping_rounds=50,
    n_estimators=500,
)
xg_step.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100,
)
```


```pycon
>>> half1 = X_test.iloc[:len(X_test) // 2]
>>> half2 = X_test.iloc[len(X_test) // 2:]
>>> pred1 = xg_step.predict_proba(half1)
>>> pred2 = xg_step.predict_proba(half2)
```

```python
# histogram of predicted probabilities for the first half and the second half
fig, ax = plt.subplots(figsize=(6, 3))
bins = 20
pd.Series(pred1[:, 1]).plot.hist(alpha=0.5, label='Half 1', bins=bins, ax=ax)
pd.Series(pred2[:, 1]).plot.hist(alpha=0.5, label='Half 2', bins=bins, ax=ax)
plt.legend()
plt.title('Histogram of Predicted Probabilities')
plt.xlabel('Predicted Probability of Class 1')
```



```pycon
>>> import scipy.stats as stats
>>> ks_statistic, p_value = stats.ks_2samp(pred1[:, 1], pred2[:, 1])
>>> print(f"KS Statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}")
```


## Training a Model to Separate Old Rows from New Rows

```python
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier


drift_X = pd.concat([X_train, X_test], axis='index', ignore_index=True)
drift_y = pd.Series(
    [0] * len(X_train) + [1] * len(X_test),
    name='is_newer_period',
)

cv = model_selection.StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)
drift_clf = xgb.XGBClassifier(random_state=42)

drift_auc = model_selection.cross_val_score(
    drift_clf,
    drift_X,
    drift_y,
    cv=cv,
    scoring='roc_auc',
)
```


```pycon
>>> print(pd.Series(drift_auc, name='drift_auc'))
>>> print(f'Mean ROC AUC: {drift_auc.mean():.3f}')

```


```pycon
>>> drift_clf.fit(drift_X, drift_y)
>>> print(
...     pd.Series(drift_clf.feature_importances_, index=drift_X.columns)
...     .sort_values(ascending=False)
... )
```


## Logging for Drift


```python
import whylogs as why

results_train = why.log(pandas=X_train
    .assign(pred=xg_step.predict(X_train),
            prob=xg_step.predict_proba(X_train)[:, 1],))
results_test = why.log(pandas=X_test
    .assign(pred=xg_step.predict(X_test),
            prob=xg_step.predict_proba(X_test)[:, 1],))
```

```python
from whylogs.viz import NotebookProfileVisualizer

viz = NotebookProfileVisualizer()
viz.set_profiles(target_profile_view=results_test.view(),
                 reference_profile_view=results_train.view())
viz.profile_summary()
```


```python
viz.summary_drift_report()
```


```python
viz.double_histogram(feature_name='prob')
```



```python
train_prof = results_train.profile()
train_prof.write("train_profile.bin")
```


```python
# read a profile from disk
prof2 = why.read("train_profile.bin")
```


```pycon
>>> prof2_df = prof2.view().to_pandas()
>>> print(prof2_df.head())
```

## Detecting Performance Drift
## Best Practices for Detecting Drift



## A Small Drift Endpoint
## Summary
## Exercises


# Regression with XGBoost


## Loading the Dirty Devil data

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from sklearn import base, linear_model, metrics, model_selection, tree
from sk_stepwise import Float, Int, StepwiseOptunaSearchCV
```


```python
DIRTYDEVIL_URL = 'https://github.com/mattharrison/datasets/raw/master/data/dirtydevil.txt'
HANKSVILLE_URL = 'https://github.com/mattharrison/datasets/raw/master/data/hanksville.csv'


def to_denver_time(df_: pd.DataFrame, time_col: str, tz_col: str) -> pd.Series:
    return (
        df_.assign(**{tz_col: df_[tz_col].replace('MDT', 'MST7MDT')})
        .groupby(tz_col)[time_col]
        .transform(
            lambda s: pd.to_datetime(s)
            .dt.tz_localize(s.name, ambiguous=True)
            .dt.tz_convert('America/Denver')
        )
    )


def tweak_river(df_: pd.DataFrame) -> pd.DataFrame:
    return (
        df_.assign(datetime=to_denver_time(df_, 'datetime', 'tz_cd'))
        .rename(columns={'144166_00060': 'cfs', '144167_00065': 'gage_height'})
        .loc[:, ['datetime', 'tz_cd', 'cfs', 'gage_height']]
    )


def tweak_temp(df_: pd.DataFrame) -> pd.DataFrame:
    return (
        df_.assign(
            DATE=pd.to_datetime(df_.DATE).dt.tz_localize('America/Denver', ambiguous=False)
        )
        .loc[:, ['DATE', 'PRCP', 'TMIN', 'TMAX', 'TOBS']]
    )


raw_river = pd.read_csv(
    DIRTYDEVIL_URL,
    skiprows=lambda num: num < 34 or num == 35,
    sep='	',
)
raw_temp = pd.read_csv(HANKSVILLE_URL)

river_df = tweak_river(raw_river)
temp_df = tweak_temp(raw_temp)

```


```pycon
>>> print(river_df)
```


```pycon
>>> print(temp_df)
```


```python
dd = (
    river_df
    .groupby(pd.Grouper(key='datetime', freq='D'))
    .median(numeric_only=True)
    .merge(temp_df, left_index=True, right_on='DATE', how='inner', validate='1:1')
    .sort_values('DATE')
)
```


```pycon
>>> print(dd)
```

## A little exploratory analysis

```python
fig, ax = plt.subplots(figsize=(6, 3))
dd.plot(x='DATE', y='cfs', ax=ax, legend=False)
ax.set_title('Dirty Devil flow over time')
ax.set_ylabel('cfs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('img/river_flow.png', bbox_inches='tight', dpi=300)
```




```python
fig, ax = plt.subplots(figsize=(6, 3))
(
    dd.query('cfs < 6000')
    .plot(x='DATE', y='cfs', ax=ax, legend=False)
)
ax.set_title('Dirty Devil flow with extreme spikes trimmed')
ax.set_ylabel('cfs')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('img/river_flow_trimmed.png', bbox_inches='tight', dpi=300)

```


```pycon
>>> print(dd
...     .assign(**{f'lag_{i}_cfs': dd.cfs.shift(i) 
...                for i in range(1, 8)})
...     .corr(numeric_only=True)
...     .loc[['cfs'], [f'lag_{i}_cfs' for i in range(1, 8)]]
...     .T
... )
```


## Feature engineering


```python
def cyclic_encode(adf: pd.DataFrame, col: str, x_suffix: str = '_x', y_suffix: str = '_y') -> pd.DataFrame:
    period = max(adf[col].nunique(), 1)
    return adf.assign(
        **{
            f'{col}{x_suffix}': np.sin((2 * np.pi * adf[col]) / period),
            f'{col}{y_suffix}': np.cos((2 * np.pi * adf[col]) / period),
        }
    )


def tweak_dirty(adf: pd.DataFrame) -> pd.DataFrame:
    cols = ['cfs', 'gage_height', 'PRCP', 'TMIN', 'TMAX', 'TOBS', 'dow',
       'day', 'month', 'week', 'dow_x', 'dow_y', 'day_x', 'day_y', 'month_x',
       'month_y', 'week_x', 'week_y', 'cfs_lag_1', 'cfs_lag_7', 'gage_lag_1',
       'rolling_cfs_7', 'rolling_prcp_7', 'target']
    out = (
        adf
        .assign(
            dow=pd.col('DATE').dt.day_of_week,
            day=pd.col('DATE').dt.day,
            month=pd.col('DATE').dt.month,
            week=pd.col('DATE').dt.isocalendar().week.astype(int),
        )
        .query('month <= 5')
        .pipe(cyclic_encode, col='dow')
        .pipe(cyclic_encode, col='day')
        .pipe(cyclic_encode, col='month')
        .pipe(cyclic_encode, col='week')
        .assign(
            cfs_lag_1=pd.col('cfs').shift(1),
            cfs_lag_7=pd.col('cfs').shift(7),
            gage_lag_1=pd.col('gage_height').shift(1),
            rolling_cfs_7=pd.col('cfs').shift(1).rolling(7).mean(),
            rolling_prcp_7=pd.col('PRCP').shift(1).rolling(7).sum(),
            target=pd.col('cfs').shift(-7),
        )
        .loc[:, cols]
    )
    return out

```


```python
# plot cyclic encoding for day of year
fig, ax = plt.subplots(figsize=(6, 4))

(dd
 .assign(day=pd.col('DATE').dt.day)
 .pipe(cyclic_encode, col='day').plot(x='day_x', y='day_y', kind='scatter', ax=ax, legend=False,
                     # color by day of year
                     s=300,
                     c='day', cmap='hsv', edgecolor='k', alpha=1                 
                                      )
)
ax.set_title('Cyclic encoding for day of month')
ax.set_xlabel('day_x')
ax.set_ylabel('day_y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.savefig('img/cyclic_encoding_day.png', bbox_inches='tight', dpi=300)
```



```pycon
>>> model_df = tweak_dirty(dd)
>>> print(model_df)
```

## Pipeline Code

```python
from sklearn import base, pipeline

class DDFeatureEngineer(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        return tweak_dirty(X).drop(columns='target')
```


```pycon
>>> fe = DDFeatureEngineer()
>>> print(fe.fit_transform(dd))
```

## Train/test split


```python
X = dd.query('not cfs.isna()')
y = dd.loc[X.index, 'cfs']

split_at = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
```


```pycon
>>> from sklearn.model_selection import TimeSeriesSplit
>>> 
>>> tscv = TimeSeriesSplit(n_splits=5)
>>> 
>>> for train_index, test_index in tscv.split(X):
...     X_train_tss, X_test_tss = X.iloc[train_index], X.iloc[test_index]
...     y_train_tss, y_test_tss = y.iloc[train_index], y.iloc[test_index]
...     print(f'Start index: {train_index[0]}, Length: {len(train_index)}')
```


## Baselines

```pycon
>>> lr = linear_model.LinearRegression()
>>> lr.fit(X_train, y_train)
```


```python
from sklearn import set_config

set_config(transform_output='pandas')

lr_pipeline = pipeline.Pipeline([
    ('fe', DDFeatureEngineer()),
    ('imputer', imputation.MeanMedianImputer()),
    ('model', linear_model.LinearRegression()),
])
```

```python
lr_pipeline.fit(X_train, y_train)
lr_pipeline.score(X_test, y_test)
```


```python
class DebugTransformer(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        print(f'Shape of X: {X.shape}')
        print(f'Columns of X: {X.columns.tolist()}')
        return X    
```


```pycon
>>> dt = DebugTransformer()
>>> print(dt.fit_transform(X_train))
```


```pycon
>>> lr_pipeline2 = pipeline.Pipeline([
...     ('debug1', DebugTransformer()),
...     ('fe', DDFeatureEngineer()),
...     ('debug2', DebugTransformer()),
...     ('lr', linear_model.LinearRegression()),
... ])
>>> lr_pipeline2.fit(X_train, y_train)
```


```python
import pandas as pd
from sklearn import base, linear_model, metrics, model_selection, tree

DIRTYDEVIL_URL = 'https://github.com/mattharrison/datasets/raw/master/data/dirtydevil.txt'
HANKSVILLE_URL = 'https://github.com/mattharrison/datasets/raw/master/data/hanksville.csv'


def to_denver_time(df_: pd.DataFrame, time_col: str, tz_col: str) -> pd.Series:
    return (
        df_.assign(**{tz_col: df_[tz_col].replace('MDT', 'MST7MDT')})
        .groupby(tz_col)[time_col]
        .transform(
            lambda s: pd.to_datetime(s)
            .dt.tz_localize(s.name, ambiguous=True)
            .dt.tz_convert('America/Denver')
        )
    )


def tweak_river(df_: pd.DataFrame) -> pd.DataFrame:
    return (
        df_.assign(datetime=to_denver_time(df_, 'datetime', 'tz_cd'))
        .rename(columns={'144166_00060': 'cfs', '144167_00065': 'gage_height'})
        .loc[:, ['datetime', 'tz_cd', 'cfs', 'gage_height']]
    )


def tweak_temp(df_: pd.DataFrame) -> pd.DataFrame:
    return (
        df_.assign(
            DATE=pd.to_datetime(df_.DATE).dt.tz_localize('America/Denver', 
                                                         ambiguous=False)
        )
        .loc[:, ['DATE', 'PRCP', 'TMIN', 'TMAX', 'TOBS']]
    )

def merge_data(river_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    return (
        river_df
        .groupby(pd.Grouper(key='datetime', freq='D'))
        .median(numeric_only=True)
        .merge(temp_df, left_index=True, right_on='DATE', how='inner', 
                validate='1:1')
        .sort_values('DATE')
        .query('DATE.dt.month <= 5') # remove rows BEFORE pipeline!
    )

raw_river = pd.read_csv(
    DIRTYDEVIL_URL,
    skiprows=lambda num: num < 34 or num == 35,
    sep='	',
)


raw_temp = pd.read_csv(HANKSVILLE_URL)
river_df = tweak_river(raw_river)
temp_df = tweak_temp(raw_temp)
merged_df = merge_data(river_df, temp_df)
```


```python
def cyclic_encode(adf: pd.DataFrame, col: str, x_suffix: str = '_x', 
                  y_suffix: str = '_y') -> pd.DataFrame:
    period = max(adf[col].nunique(), 1)
    return adf.assign(
        **{
            f'{col}{x_suffix}': np.sin((2 * np.pi * adf[col]) / period),
            f'{col}{y_suffix}': np.cos((2 * np.pi * adf[col]) / period),
        }
    )


def tweak_dirty(adf: pd.DataFrame) -> pd.DataFrame:
    cols = ['cfs', 'gage_height', 'PRCP', 'TMIN', 'TMAX', 'TOBS', 'dow',
       'day', 'month', 'week', 'dow_x', 'dow_y', 'day_x', 'day_y', 'month_x',
       'month_y', 'week_x', 'week_y', 'cfs_lag_1', 'cfs_lag_7', 'gage_lag_1',
       'rolling_cfs_7', 'rolling_prcp_7', 'target']
    out = (
        adf
        .assign(
            dow=pd.col('DATE').dt.day_of_week,
            day=pd.col('DATE').dt.day,
            month=pd.col('DATE').dt.month,
            week=pd.col('DATE').dt.isocalendar().week.astype(int),
        )
        #.query('month <= 5') # REMOVED!!!
        .pipe(cyclic_encode, col='dow')
        .pipe(cyclic_encode, col='day')
        .pipe(cyclic_encode, col='month')
        .pipe(cyclic_encode, col='week')
        .assign(
            cfs_lag_1=pd.col('cfs').shift(1),
            cfs_lag_7=pd.col('cfs').shift(7),
            gage_lag_1=pd.col('gage_height').shift(1),
            rolling_cfs_7=pd.col('cfs').shift(1).rolling(7).mean(),
            rolling_prcp_7=pd.col('PRCP').shift(1).rolling(7).sum(),
            target=pd.col('cfs').shift(-7),
        )
        .loc[:, cols]
    )
    return out


class DDFeatureEngineer(base.BaseEstimator, base.TransformerMixin):
    
    def fit(self, X, y=None):
        self.fitted_ = True
        return self
    
    def transform(self, X):
        return tweak_dirty(X).drop(columns='target')
```


```python
X = merged_df.query('not cfs.isna()')
#y = merged_df.loc[X.index, 'target']
y = (merged_df
    .assign(target=merged_df.cfs.shift(-7))
    .loc[X.index, 'target']
    .dropna()
    )
# because y might have had some rows dropped, we need to make sure to align it with X
X = X.loc[y.index]

split_at = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
```


```pycon
>>> lr_pipeline3 = pipeline.Pipeline([
...     ('fe', DDFeatureEngineer()),
...     ('median_imputer', imputation.MeanMedianImputer(imputation_method='median')),
...     ('lr', linear_model.LinearRegression()),
... ])
... 
>>> lr_pipeline3.fit(X_train, y_train)
>>> lr_pipeline3.score(X_test, y_test)
```



```pycon
>>> dt_pipeline = pipeline.Pipeline([
...     ('fe', DDFeatureEngineer()),
...     ('median_imputer', imputation.MeanMedianImputer(imputation_method='median')),
...     ('dt', tree.DecisionTreeRegressor(random_state=42)),
... ])
>>> dt_pipeline.fit(X_train, y_train)
>>> dt_pipeline.score(X_test, y_test)
```


```pycon
>>> xgb_pipeline = pipeline.Pipeline([
...     ('fe', DDFeatureEngineer()),
...     ('xgb', xgb.XGBRegressor(random_state=42)),
... ])
>>> xgb_pipeline.fit(X_train, y_train)
>>> xgb_pipeline.score(X_test, y_test)
```


```pycon
>>> xgb_pipeline.score(X_train, y_train)
```


## Metrics




```python
results = {}
for metric in ['r2_score', 'mean_absolute_error', 'mean_squared_error',
               'root_mean_squared_error',
               'median_absolute_error', 'max_error']:
    fn = getattr(metrics, metric)
    results[metric] = {
        'Linear Regression': fn(y_test, lr_pipeline3.predict(X_test)),
        'Decision Tree': fn(y_test, dt_pipeline.predict(X_test)),
        'XGBoost': fn(y_test, xgb_pipeline.predict(X_test)),
    }
```

```pycon
>>> print(pd.DataFrame(results).T)
```


## Prediction error

```python
fig, ax = plt.subplots(figsize=(6, 3))
# plot prediction error

(y_train
 .to_frame()
 .assign(pred=lr_pipeline3.predict(X_train),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Train', alpha=0.5,
           color='blue'))

(y_test
 .to_frame()
 .assign(pred=lr_pipeline3.predict(X_test),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Test', alpha=0.5, 
       color='orange')) 

ax.set_title('Residual plot for Linear Regression')
fig.savefig('img/residual_plot_lr.png', bbox_inches='tight', dpi=300)
```


```python
# residual plot for decision tree
fig, ax = plt.subplots(figsize=(6, 3))
(y_train
 .to_frame()
 .assign(pred=dt_pipeline.predict(X_train),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Train', alpha=0.5,
           color='blue'))       
(y_test
    .to_frame()
    .assign(pred=dt_pipeline.predict(X_test),
            residual=pd.col('target') - pd.col('pred'))
    .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Test', alpha=0.5, color='orange')) 
ax.set_title('Residual plot for Decision Tree')
fig.savefig('img/residual_plot_dt.png', bbox_inches='tight', dpi=300)
```



```python
# xgboost residual plot
fig, ax = plt.subplots(figsize=(6, 3))
(y_train
 .to_frame()
 .assign(pred=xgb_pipeline.predict(X_train),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Train', alpha=0.5,
           color='blue'))
(y_test
 .to_frame()
 .assign(pred=xgb_pipeline.predict(X_test),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Test', alpha=0.5, color='orange')) 
ax.set_title('Residual plot for XGBoost')
fig.savefig('img/residual_plot_xgb.png', bbox_inches='tight', dpi=300)
```


## Tuning with `sk-stepwise`


```python
fe = DDFeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
X_test_fe = fe.transform(X_test)    

base_xgb = xgb.XGBRegressor(
    n_estimators=500,
    objective='reg:squarederror',
    random_state=42,
)

search = StepwiseOptunaSearchCV(
    estimator=base_xgb,
    param_distributions=[
        {
            'max_depth': Int(2, 6),
            'min_child_weight': Int(2, 12),
        },
        {
            'subsample': Float(0.7, 1.0),
            'colsample_bytree': Float(0.6, 1.0),
        },
        {
            'reg_lambda': Float(1.0, 12.0, log=True),
            'gamma': Float(0.0, 0.5),
        },
        {
            'learning_rate': Float(0.01, 0.1, log=True),
        },
    ],
    n_trials_per_step=20,
    cv=model_selection.TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    random_state=42,
)

search.fit(X_train_fe, y_train)
search.best_params_
```

```python
step_rows = []
for idx, step in enumerate(search.step_results_, start=1):
    row = {
        'step': idx,
        'best_value': step['best_value'],
        'n_trials': step['n_trials'],
        'improvement': step['improvement'],
    }
    row.update(step['best_params'])
    step_rows.append(row)

print(pd.DataFrame(step_rows))
```


```python
params = {'max_depth': 2,
 'min_child_weight': 12,
 'subsample': 0.9639652156387968,
 'colsample_bytree': 0.642731263057218,
 'reg_lambda': 10.470649776146537,
 'gamma': 0.38477083439279275,
 'learning_rate': 0.010042665707938011}

xgb_step = xgb.XGBRegressor(**params, random_state=42, n_estimators=500,
                 early_stopping_rounds=20)
xgb_step.fit(X_train_fe, y_train, eval_set=[(X_test_fe, y_test)], verbose=False)


compare =pd.DataFrame({
    'XGB basic': 
        {'mae': metrics.mean_absolute_error(y_test, xgb_pipeline.predict(X_test)),
        'rmse': metrics.mean_squared_error(y_test, xgb_pipeline.predict(X_test)),
        'median_ae': metrics.median_absolute_error(y_test, xgb_pipeline.predict(X_test)),
        'max_error': metrics.max_error(y_test, xgb_pipeline.predict(X_test)),
        'r2_score': metrics.r2_score(y_test, xgb_pipeline.predict(X_test))},
    'XGB stepwise': 
        {'mae': metrics.mean_absolute_error(y_test, xgb_step.predict(X_test_fe)),
        'rmse': metrics.mean_squared_error(y_test, xgb_step.predict(X_test_fe)),
        'median_ae': metrics.median_absolute_error(y_test, xgb_step.predict(X_test_fe)),
        'max_error': metrics.max_error(y_test, xgb_step.predict(X_test_fe)),
        'r2_score': metrics.r2_score(y_test, xgb_step.predict(X_test_fe))},
    }
)
```


```pycon
>>> print('Test scores for default and tuned XGBoost models:')
>>> print(compare.round(2).T)
```


```python
compare_training = pd.DataFrame({
    'XGB basic': {
        'mae': metrics.mean_absolute_error(y_train, xgb_pipeline.predict(X_train)),
        'rmse': metrics.mean_squared_error(y_train, xgb_pipeline.predict(X_train)),
        'median_ae': metrics.median_absolute_error(y_train, xgb_pipeline.predict(X_train)),
        'max_error': metrics.max_error(y_train, xgb_pipeline.predict(X_train)),
        'r2_score': metrics.r2_score(y_train, xgb_pipeline.predict(X_train))},
    'XGB stepwise': {
        'mae': metrics.mean_absolute_error(y_train, xgb_step.predict(X_train_fe)),
        'rmse': metrics.mean_squared_error(y_train, xgb_step.predict(X_train_fe)),
        'median_ae': metrics.median_absolute_error(y_train, xgb_step.predict(X_train_fe)),
        'max_error': metrics.max_error(y_train, xgb_step.predict(X_train_fe)),
        'r2_score': metrics.r2_score(y_train, xgb_step.predict(X_train_fe))},
}
)   
```


```pycon
>>> print('Training scores for default and tuned XGBoost models:')
>>> print(compare_training.round(2).T)
```



```python
# residual plot for xgboost
fig, ax = plt.subplots(figsize=(6, 3))
(y_train
 .to_frame()
 .assign(pred=xgb_step.predict(X_train_fe),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Train', alpha=0.5,
           color='blue'))
(y_test
 .to_frame()
 .assign(pred=xgb_step.predict(X_test_fe),
         residual=pd.col('target') - pd.col('pred'))
 .plot(x='pred', y='residual', ax=ax, kind='scatter', label='Test', alpha=0.5, color='orange')) 
ax.set_title('Residual plot for XGBoost after tuning')
fig.savefig('img/residual_plot_xgb_tuned.png', bbox_inches='tight', dpi=300)
```



## Interpretation

```python
fig, ax = plt.subplots(figsize=(6, 3))
(pd.Series(xgb_step.feature_importances_, index=X_train_fe.columns)
    .sort_values()
    .tail(12)
    .plot.barh(ax=ax)
)
ax.set_title('XGBoost feature importance')
```



```python
import treefi

tree_fe = treefi.feature_importance(xgb_step)
```

```pycon
>>> print(tree_fe.sort_values('gain'))
```

## Feature interactions

```python
tree_fi = treefi.feature_interactions(xgb_step)
```

```pycon
>>> print(tree_fi
...    .sort_values('gain', ascending=False)
...    .query('interaction_order > 1')
...    )
```


```python
fig, ax = plt.subplots(figsize=(6, 4))

# plot train values against TOBS vs cfs
pd.DataFrame({
    'pred': xgb_step.predict(X_train_fe),
    'TOBS': X_train_fe.TOBS,
    'cfs': X_train_fe.cfs,}).plot(x='TOBS', y='cfs', kind='scatter', ax=ax, alpha=0.5, color='blue', label='Train')  
ax.set_title('cfs vs TOBS')


# plot test values against TOBS vs cfs
pd.DataFrame({
    'pred': xgb_step.predict(X_test_fe),
    'TOBS': X_test_fe.TOBS,
    'cfs': X_test_fe.cfs,
}).plot(x='TOBS', y='cfs', kind='scatter', ax=ax, alpha=0.9, color='orange', label='Test')
ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax.set_ylim(0, 300)
fig.savefig('img/cfs_vs_tobs.png', bbox_inches='tight', dpi=300)
```



## SHAP 


```python
explainer = shap.TreeExplainer(xgb_step, X_test_fe)
shap_values = explainer(X_test_fe)
```




```pycon
>>> explainer.expected_value
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
res = shap.plots.beeswarm(shap_values)
```



```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(shap_values[:, 'cfs'], ax=ax)
ax.set_title('SHAP values for cfs feature')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(shap_values[:, 'cfs'], color=shap_values, ax=ax)
ax.set_title('SHAP values for cfs feature')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(shap_values[:, 'cfs'], color=shap_values[:, 'TOBS'], ax=ax)
ax.set_title('SHAP values for cfs feature')
```


```python
fig, ax = plt.subplots(figsize=(6, 3))
shap.plots.scatter(shap_values[:, 'TOBS'], color=shap_values[:, 'rolling_cfs_7'], ax=ax)
ax.set_title('SHAP values for TOBS feature')
```



## Summary
## Exercises

# Optimizing Regression with AutoXGB
## The Karpathy AutoResearch Framing
## Brainstorming 


## My Loop


## A Metric

## The Regression Problem
## What the AI Did Well

## Where the Model Family Changed

## Error Analysis

## The Temptation to Believe the Curve


## What the Human Was Good For
## Why Reproducibility is Important
## What I Learned from AutoXGB
## Hints for Using AI

## Summary
## Exercises

