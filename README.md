# `riskmodels` - 风险模型工具库

`riskmodels`意在提供风险模型开发中常用的函数和算法，目前主要覆盖了以下几类功能：

* 数据探索
* 变量分箱
* 逻辑回归建模
* 评分卡转换
* 模型评估

以下对主要功能进行说明，详细说明请参见代码文档。

## 数据探索

### 样本分布: `riskmodels.utils.sample_stats`

该函数作用是统计样本中的总样本数、好坏样本数及坏率，一般会结合 groupby 使用。如下例：

```{python}
# 按照申请月份进行样本统计
df.groupby('apply_month').apply(sample_stats, target='y')

# 按照不同信贷产品进行样本统计
df.groupby('product_id').apply(sample_stats, target='y')
```

### 变量探索: `riskmodels.detector.detect`

> 注：该函数源自 `toad`

该函数用于变量分布。对于数值型变量，统计其空值率、最大值、最小值、平均值、方差等统计量；对于类别型变量，统计器出现频次最高的类别。

## 变量分箱: `riskmodels.scorecard`模块

本模块基于 `scorecardpy` 项目进行重构，主要目的是提供分箱方法的可扩展性。

原项目的分箱步骤为：特殊值处理 → 细分箱：等距分箱 → 粗分箱：ChiMerge/树方法，本次重构进行了如下优化：

* 细分箱增加等频分箱，由于信贷场景的数据偏度极大，等距分箱可能在数据集中部分丢失细节，等频分箱更为合适
* 粗分箱中的树方法，增加了对单调性约束的支持(通过`ensure_monotonic=True`打开，默认为`False`)

### `woebin`函数

```{python}
def woebin(dt,
           y,
           x=None,
           var_skip=None,
           breaks_list=None,
           special_values=None,
           positive="bad|1",
           no_cores=None,
           methods=None,
           ignore_const_cols=True,
           ignore_datetime_cols=True,
           check_cate_num=True,
           replace_blank=True,
           **kwargs): ...
```

该函数与`sc.woebin`函数接口基本类似，主要变更如下：

* methods: 默认为`['quantile', 'tree']`, 即采用等频分箱→树分箱的分箱方式；该参数默认可支持的分箱方法包括
  * `hist`: 等距分箱，注册类`riskmodels.scorecard.HistogramInitBin`
  * `quantile`: 等频分箱，注册类`riskmodels.scorecard.QuantileInitBin`
  * `tree`: 树分箱，注册类`riskmodels.scorecard.TreeOptimBin`
  * `chi2`/`chimerge`: ChiMerge分箱，注册类`riskmodels.scorecard.ChiMergeOptimBin`
  使用该参数有以下**注意事项**：
  * 首个分箱方法必须为（无监督）细分箱方法，此处可选为`hist`和`quantile`两类；
  * 细分箱方法不可位于其他分箱方法之后，如`['quantile', 'tree']`，此时等频分箱方法不生效；
  * 可以只包含细分箱，如`['quantile']`或`['hist']`，此时为纯无监督分箱；
  * 列表长度可以大于2，例如：`['quantile', 'tree', 'chi2]`，即在树分箱的基础上，再用ChiMerge方法对无显著差异的相邻分箱进行合并。
* `**kwargs`: 该参数为各个分箱方法所需要的参数，具体可见分箱方法类的文档，下列最常见参数。
  * 等距分箱和等频分箱
    * initial_bins: 细分箱的数量，默认20
  * 树分箱和ChiMerge分箱
    * bin_num_limit: 最终分箱的最大数量（不含特殊值），默认5
    * count_distr_limit: 分箱样本占总样本的最小比例，默认0.05
    * stop_limit: 分箱停止条件，树分箱为IV值相对增量，ChiMerge为独立性检验P值，默认0.05
    * ensure_monotonic（仅树分箱支持）: 是否保证单调性（不含特殊值），默认`False`

#### 分箱方法的扩展

（略）

### `woebin_ply`函数

```{python}
def woebin_ply(dt, bins, no_cores=None, replace_blank=False, value='woe'):
    ...
```

该函数与`sc.woebin_ply`函数接口基本类似，增加如下参数：

* value: 可选项为 `['woe', 'index', 'bin']`，默认为 'woe'
  * value='woe'时，将原始值替换为woe值，返回的字段名为 `变量名_woe`，与`sc.woebin_ply`一致；
  * value='index'时，将原始值替换为变量分箱结果数据框中的index，返回的字段名为 `变量名_index`；
  * value='bin' 时，返回结果为分箱区间 [a,b) 【数值型变量】或 a%,%b 【类别型变量】，返回的字段名为 `变量名_bin`。

### `woebin_psi`函数

```{python}
def woebin_psi(df_base, df_cmp, bins):
    ...
```

该函数为新增函数，用于计算变量PSI值，详细使用方式见函数文档。





### 其他函数

其他常用函数列举如下：

* sc_bins_to_df: 整合`woebin`返回值，生成woe表和iv表
* woebin_breaks: 根据`woebin`返回值保存切分点和特殊值点
* woebin_plot: 根据`woebin`返回值生成bivar图像


## 逻辑回归建模

## 评分卡转换

### `make_scorecard`函数

```{python}
def make_scorecard(sc_bins, coef, *, base_points=600, base_odds=50, pdo=20):
    ...
```

该函数用于生成评分卡，其中`coef`为各个入模变量的系数字典： `{变量名_woe: 系数}`

## 模型评估