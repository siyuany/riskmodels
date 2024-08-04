# -*- encoding: utf-8 -*-
"""
scorecard.py - 变量分箱算法

本模块从 scorecardpy(https://github.com/ShichenXie/scorecardpy)移植，并对代码进行了
大量重构，以提供更好的扩展性和向后兼容性。本模块主要接口尽可能保持与 scorecardpy 相同，主要有
如下接口函数：

* woebin
* woebin_ply
* woebin_plot

此外，提供以下接口函数：

* sc_bins_to_df
* make_scorecard
* woebin_breaks
* woebin_psi
"""
import itertools
import multiprocessing as mp
import re
import time
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from scipy.stats import chi2
from scipy.stats import chi2_contingency

import syriskmodels.logging as logging
from syriskmodels.evaluate import psi
from syriskmodels.utils import monotonic
from syriskmodels.utils import round_
from syriskmodels.utils import str_to_list

OK = 0
CONST = 10
TOO_MANY_VALUES = 20


def check_uniques(s: pd.Series, max_cate_num: int = 50):
  """
  检查变量非空单一值数，当非空单一值数小于等于1时，该变量为常量，不适合用于建模；当变量为类别变量（非数值型）时，
  过多类别值也不适合用于建模。对满足上述情况的变量，本函数返回 False；否则返回 True。
  Args:
    s: pd.Series
    max_cate_num: int, 最大允许类别数

  Returns: bool

  """
  n_uniques = len(np.unique(s[~s.isna()]))
  if n_uniques <= 1:
    return CONST
  elif (not is_numeric_dtype(s)) and n_uniques > max_cate_num:
    return TOO_MANY_VALUES
  else:
    return OK


def replace_blank_string(s: pd.Series):
  return s.replace('', np.nan)


def check_y(dat: pd.DataFrame, y: str, *, positive: Union[int, float] = 1):
  """
  对dat中y标进行如下检查：(1)dat[y]是否为数值型；(2)dat[y]空值检查；(3)dat[y]是否为(0,1)二值型变量.
  完成检查后，将dat[y]等于positive的记录记为1（正样本），否则记为0.
  """
  dat = dat.copy()
  try:
    if not is_numeric_dtype(dat[y]):
      msg = f'The target column {y} is not numeric type, dtype={dat[y].dtypes}'
      logging.error(msg)
      raise TypeError(msg)

    if dat[y].isna().any():
      logging.warn(f'There are NaNs in {y} column. '
                   f'The corresponding records are removed.')
      dat = dat.dropna(subset=[y])

    if dat[y].nunique() != 2:
      logging.error(f'The target column {y} is not binary.')
      raise ValueError(f'The target column {y} is not binary.')

    if np.all(dat[y] != positive):
      logging.error(f'positive value ({positive}) not in the target column.')
      raise ValueError(f'positive value ({positive}) not in the target column.')

    dat[y] = np.where(dat[y] == positive, 1, 0)

  except KeyError as err:
    logging.error(
        "Incorrect inputs; there is no \'{}\' column in dat.".format(y))
    raise KeyError(y) from err

  return dat


def x_variable(dat, y, x, var_skip=None):
  y = str_to_list(y)
  if var_skip is not None:
    y = y + str_to_list(var_skip)
  x_all = list(set(dat.columns) - set(y))

  if x is None:
    x = x_all
  else:
    x = str_to_list(x)

    if any([i in list(x_all) for i in x]) is False:
      x = x_all
    else:
      x_not_in_x_all = set(x).difference(x_all)
      if len(x_not_in_x_all) > 0:
        logging.warn("Incorrect inputs; there are {} x variables are not exist "
                     "in input data, which are removed from x. \n({})".format(
                         len(x_not_in_x_all), ', '.join(x_not_in_x_all)))
        x = set(x).intersection(x_all)

  return list(x)


def check_breaks_list(breaks_list) -> dict:
  if breaks_list is not None:
    # is string
    if isinstance(breaks_list, str):
      breaks_list = eval(breaks_list)
    # is not dict
    if not isinstance(breaks_list, dict):
      raise Exception("Incorrect inputs; breaks_list should be a dict.")
  else:
    breaks_list = {}
  return breaks_list


def check_special_values(special_values, xs) -> dict:
  if special_values is not None:
    # # is string
    # if isinstance(special_values, str):
    #     special_values = eval(special_values)
    if isinstance(special_values, list):
      logging.warn(
          "The special_values should be a dict. Make sure special values "
          "are exactly the same in all variables if special_values is "
          "a list.")
      sv_dict = {}
      for i in xs:
        sv_dict[i] = special_values
      special_values = sv_dict
    elif not isinstance(special_values, dict):
      raise Exception(
          "Incorrect inputs; special_values should be a list or dict.")
  else:
    special_values = {}
  return special_values


def woebin(
    dt,
    y,
    x=None,
    var_skip=None,
    breaks_list=None,
    special_values=None,
    # positive="bad|1",
    positive=1,
    no_cores=None,
    methods=None,
    max_cate_num=50,
    replace_blank=np.nan,
    **kwargs):
  if methods is None:
    methods = ['quantile', 'tree']

  # start time
  start_time = time.time()
  # arguments
  dt = dt.copy(deep=True)
  y = str_to_list(y)
  x = str_to_list(x)
  if x is not None:
    dt = dt[y + x]
  # check y
  dt = check_y(dt, y[0], positive=positive)

  # x variable names
  xs = x_variable(dt, y, x, var_skip)
  xs_len = len(xs)
  # breaks_list
  breaks_list = check_breaks_list(breaks_list)
  # special_values
  special_values = check_special_values(special_values, xs)

  # binning for each x variable
  # loop on xs
  if (no_cores is None) or (no_cores < 1):
    all_cores = mp.cpu_count() - 1
    no_cores = int(
        np.ceil(xs_len / 5 if xs_len / 5 < all_cores else all_cores * 0.9))

  # y list to str
  y = y[0]

  woe_bin = WOEBinFactory.build(methods, **kwargs)

  tasks = [
      (
          # dtm definition
          pd.DataFrame({
              'y': dt[y],
              'variable': x_i,
              'value': dt[x_i]
          }),
          # breaks_list
          breaks_list.get(x_i),
          # special_values
          special_values.get(x_i),
          max_cate_num,
          replace_blank,
      ) for x_i in xs
  ]

  logging.info(f'开始分箱，特征数 {len(tasks)}，样本数 {len(dt)}')

  if no_cores == 1:
    bins = dict(zip(xs, itertools.starmap(woe_bin, tasks)))
  else:
    pool = mp.Pool(processes=no_cores)
    bins = dict(zip(xs, pool.starmap(woe_bin, tasks)))
    pool.close()

  # running time
  running_time = time.time() - start_time
  logging.info('分箱完成：Binning on {} rows and {} columns in {}'.format(
      dt.shape[0], len(xs), time.strftime("%H:%M:%S",
                                          time.gmtime(running_time))))

  return bins


class WOEBinFactory(object):
  """WOEBin工厂方法类，与ComposedWOEBin配合，提供WOEBin子类的组装功能。

    本模块原生提供了两种细分箱方法：QuantileInitBin与HistogramInitBin，两种粗分箱方法：
    ChiMergeOptimBin与TreeOptimBin。一般在实际使用时需要将细分箱与粗分箱结合使用，例如：
    先用QuantileInitBin进行等频分箱，然后再用TreeOptimBin对进行细分箱。

    实现上述功能的代码如下：

    >>> woebin = WOEBinFactory.build(['quantile', 'tree'])
    >>> woebin(dtm)

    其中，quantile和tree为上述两个类的注册名，也可显示写出类名如下：

    >>> woebin = WOEBinFactory.build([QuantileInitBin, TreeOptimBin])
    >>> woebin(dtm)

    详情请见`WOEBinFactory.register`和`WOEBinFactory.build`方法文档。
    """
  __woebin_class_mapping = {}

  @classmethod
  def register(cls, names):
    """
        注册分箱类的装饰器。对分箱类使用该装饰器并指定注册名称后，在`build`方法中就可以使用
        注册名称替代类名。使用方法可参考`QuantileInitBin`, `HistogramInitBin`,
        `TreeOptimBin`以及`ChiMergeOptimBin`四个分箱类。

        Args:
            names: str or list[str]，分箱类的注册名称

        Returns:
            该装饰器装饰的WOEBin子类

        """
    names = str_to_list(names)

    def wrapped(bin_class):
      if not issubclass(bin_class, WOEBin):
        raise TypeError(f'类{bin_class}不是WOEBin子类，无法注册')

      for name in names:
        if name in cls.__woebin_class_mapping.keys():
          raise KeyError(f'名称{name}已存在，'
                         f'类{bin_class.__name__}不能注册为{name}')
        else:
          cls.__woebin_class_mapping[name] = bin_class

      return bin_class

    return wrapped

  @classmethod
  def get_binner(cls, bin_class, **kwargs):
    if isinstance(bin_class, str):
      try:
        bin_class = cls.__woebin_class_mapping[bin_class]
      except KeyError:
        raise KeyError(f'方法{bin_class}未注册！')

    # if issubclass(bin_class, WOEBin):
    #     binner = bin_class(**kwargs)
    # elif isinstance(bin_class, WOEBin):
    #     binner = bin_class
    if isinstance(bin_class, WOEBin):
      binner = bin_class
    elif isinstance(bin_class, type) and issubclass(bin_class, WOEBin):
      binner = bin_class(**kwargs)
    else:
      raise TypeError(f'类{bin_class}不是WOEBin实例或子类')

    return binner

  @classmethod
  def build(cls, bin_classes, **kwargs):
    """
        将多个WOEBin*按顺序*组装为一个WOEBin子类（ComposedWOEBin）,该方法可接受任意关键字
        参数，并传递给子类初始化方法。通常的组装方式为细分箱类(*InitBin) +
        粗分箱类(*OptimBin)。

        >>> woe_bin = WOEBinFactory.build (['quantile', 'tree'],
        ...                               initial_bins=20,
        ...                               bin_num_limit=8,
        ...                               min_iv_inc=0.1,
        ...                               count_distr_limit=0.05)
        >>> woe_bin
        <syriskmodels.scorecard.ComposedWOEBin object at 0x1009776a0>

        Args:
            bin_classes: WOEBin子类或子类注册名列表
            **kwargs: 子类初始化的关键字参数

        Returns:
            ComposedWOEBin实例

        """
    bin_objects = [cls.get_binner(bin_cls, **kwargs) for bin_cls in bin_classes]
    return ComposedWOEBin(bin_objects, **kwargs)


class WOEBin(object):
  """WOEBin: 对单个变量进行分箱操作的基类，所有分箱操作的类都继承本类。

    分箱类型包括细分箱(*InitBin)、粗分箱(*OptimBin)。其中细分箱用来初始化分箱，一般是通
    过等频(quantile)或等宽(histogram)的方式进行分箱；粗分箱在粗分箱结果的基础上进行，用
    来简化分箱结果，确保分箱结果的显著性和稳定性。

    WOEBin实例为可调用(callable)对象，接受dtm、breaks、special_values三个参数，当
    breaks 不为空时，WOEBin实例将直接使用breaks提供的切分点进行分箱，不再调用`woebin`方法；
    当 special_values 不为空时，调用WOEBin实例会把原数据集拆分为特殊值、非特殊值两部分，
    对特殊值部分，每个特殊值将单独成箱，不做分箱合并，对于非特殊值按照breaks参数或调用woebin
    方法返回的切分点进行分箱。注意：当原数据集中包含空值时，空值将作为特殊值单独成箱。

    `woebin`方法为分箱主方法，接收数据集中非特殊值（排除空值和special_values中指定的其他
    特殊值）部分作为入参，并返回*切分点*。扩展本类时，需要重写本方法。对于细分箱(*InitBin)，
    该方法不接受breaks参数；对于粗分箱(*OptimBin)，需要为本方法传入细分箱`woebin`方法返回
    的切分点作为breaks入参。

    分箱名约定：当x为数值型变量时，分箱名为 [a,b) 左闭右开区间；当x为类别变量时，分箱名为
    c1%,%c2%,%...%,%cn，即用%,%拼接的类别名。

    切分点约定：当x为数值型变量，分箱名为 [a,b) 时，切分点为区间右边界 b；当x为类别变量时，
    切分点与分享名相同。

    * 常用变量约定

    dtm定义:
        下述所有dtm变量为（至少）包含 variable（变量名）、y（目标变量）、value（解释
        变量值）三列的数据框（pd.DataFrame）。其中value可包含空值和其他特殊值

    ns_dtm：不含空值、特殊值的dtm（ns for non-special）

    binning_*：
        分箱信息统计表，包含 variable（变量名）、bin_chr（分箱名）、good（好样本数）、
        bad（坏样本数）四列的数据框（pd.DataFrame）。

    Args
        eps: 若某一分箱中好样本或坏样本数为0，计算woe及iv时用eps替换0，默认值0.5
    """

  def __init__(self, eps=0.5, **kwargs):
    self.epsilon = eps
    self.kwargs = kwargs

  @staticmethod
  def add_missing_spl_val(dtm: pd.DataFrame, spl_val):
    """如果数据集中存在空值，则将 'missing' 加入到special_values中"""
    special_values = spl_val
    if dtm['value'].isnull().any():
      if spl_val is None:
        special_values = ['missing']
      elif 'missing' not in spl_val:
        special_values = ['missing'] + spl_val

    return special_values

  @staticmethod
  def split_vec_to_df(vec):
    """
        特殊值/断点列表转DataFrame, 如下例所示：

        >>> vec
        ['missing', '1', '2', '3%,%4']
        >>> WOEBin.split_vec_to_df(vec)
           bin_chr  rowid value
        0  missing      0   NaN
        1        1      1     1
        2        2      2     2
        3    3%,%4      3     3
        4    3%,%4      3     4

        Args:
            vec: 特殊值列表

        Returns:
            pd.DataFrame，包含如下四列 'bin_char', 'rowid', 'value'

        """
    assert vec is not None, 'vec cannot be None'
    vec = [str(i) for i in vec]
    a = pd.DataFrame({'bin_chr': vec}).assign(rowid=lambda x: x.index)
    b = pd.DataFrame([i.split('%,%') for i in vec], index=vec) \
      .stack().replace('missing', np.nan) \
      .reset_index(name='value') \
      .rename(columns={'level_0': 'bin_chr'})[['bin_chr', 'value']]

    df = pd.merge(a, b, on='bin_chr')
    return df

  @classmethod
  def split_special_values(cls, dtm, spl_val):
    dtm['idx'] = dtm.index
    spl_val = cls.add_missing_spl_val(dtm, spl_val)
    if spl_val is not None:
      sv_df = cls.split_vec_to_df(spl_val)
      # value
      if is_numeric_dtype(dtm['value']):
        sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
        # TODO: 此处是否必要？？
        sv_df['bin_chr'] = np.where(
            np.isnan(sv_df['value']), sv_df['bin_chr'],
            sv_df['value'].astype(str))
      # dtm_sv & dtm
      dtm_merge = pd.merge(
          dtm.fillna("missing"),
          sv_df[['value', 'rowid']].fillna("missing"),
          how='left',
          on='value')
      dtm_sv = dtm_merge[~dtm_merge['rowid'].isna()][
          dtm.columns.tolist()].reset_index(drop=True)
      dtm_ns = dtm_merge[dtm_merge['rowid'].isna()][
          dtm.columns.tolist()].reset_index(drop=True)
      if len(dtm_ns) == 0:
        dtm_ns = None
      else:
        dtm_ns['value'] = dtm_ns['value'].astype(dtm['value'].dtypes)

      if dtm_sv.shape[0] == 0:
        dtm_sv = None
      else:
        dtm_sv = pd.merge(
            dtm_sv.fillna('missing'), sv_df.fillna('missing'), on='value')
    else:
      dtm_sv = None
      dtm_ns = dtm.copy()

    if dtm_sv is not None:
      dtm_sv.set_index(dtm_sv['idx'], drop=True, inplace=True)

    if dtm_ns is not None:
      dtm_ns.set_index(dtm_ns['idx'], drop=True, inplace=True)

    return {'dtm_sv': dtm_sv, 'dtm_ns': dtm_ns}

  @classmethod
  def dtm_binning_sv(cls, dtm, spl_val):
    """
    将原数据集拆分为特殊值数据集、非特殊值数据集，特殊值列表有spl_val参数给出。

    Specifications:
    1、当dtm['value']为数值类型时，每个特殊数值会单独成为一箱。例如：
    >>> vec = ['missing', '-9999%,%-999']
    如果dtm['value']为数值类型，在此函数中，-9999和-999会分开成为两箱；如果dtm['value']为
    非数值类型，则-9999和-999会合并到同一分箱。
    2、如果spl_val为None，或者dtm['value']中不存在特殊值，则返回的binning_sv为None。
    3、如果dtm['value']全为spl_val中的特殊值，则返回的dtm为None。
    4、如果入参breaks中包含'missing'，则spl_val中将不包含'missing'。

    Args:
        dtm: 应包含['y', 'variable', 'value']三列的DataFrame，详见WEOBin类文档
        spl_val: 特殊值列表

    Returns:
        {'binning_sv': 特殊值分箱统计结果, 'ns_dtm': dtm中除去特殊值外的部分}

    """
    split_dtm = cls.split_special_values(dtm, spl_val)

    dtm_sv = split_dtm['dtm_sv']
    dtm_ns = split_dtm['dtm_ns']

    if dtm_sv is None or dtm_sv.shape[0] == 0:
      binning_sv = None
    else:
      binning_sv = cls.binning(dtm_sv, dtm_sv['bin_chr'])

    return {'binning_sv': binning_sv, 'ns_dtm': dtm_ns}

  def __call__(self,
               dtm,
               breaks=None,
               special_values=None,
               max_cate_num=50,
               replace_blank=np.nan):
    dtm['value'] = replace_blank_string(dtm['value'])
    ret = check_uniques(dtm['value'], max_cate_num)
    if ret == CONST:
      return 'CONST'
    elif ret == TOO_MANY_VALUES:
      return 'TOO_MANY_VALUES'
    else:
      binning_split = self.dtm_binning_sv(dtm, special_values)
      binning_sv = binning_split['binning_sv']
      dtm = binning_split['ns_dtm']

      if dtm is None:
        binning_dtm = None
      else:
        if breaks is not None:
          binning_dtm = self.binning_breaks(dtm, breaks)
        else:
          breaks = self.woebin(dtm)
          binning_dtm = self.binning_breaks(dtm, breaks)

      bin_list = {'binning_sv': binning_sv, 'binning': binning_dtm}
      binning = pd.concat(bin_list, keys=bin_list.keys())
      binning = binning.reset_index()
      binning = binning.assign(is_sv=lambda x: x.level_0 == 'binning_sv')

      return self.binning_format(binning)

  def woebin(self, dtm, breaks=None):
    """"""
    raise NotImplementedError

  @classmethod
  def binning(cls, dtm, bin_chr):
    """
    给定dtm、分箱名序列生成binning，dtm和binning的定义见WOEBin类文档。bin_chr为
    pd.Series，长度与dtm相同，对应dtm中每个样本所属的分箱名称。

    Args:
      dtm: dtm定义见`WOEBin`类文档
      bin_chr: dtm中每个样本对应的分箱名称，pd.Series。

    Returns: binning，定义见`WOEBin`类文档
    """

    def _n0(x):
      return np.sum(x == 0)

    def _n1(x):
      return np.sum(x == 1)

    bin_chr = bin_chr.rename(index='bin_chr')
    binning = dtm.groupby(['variable', bin_chr], observed=False)['y'].agg(
        good=_n0, bad=_n1)
    binning = binning.reset_index()

    return binning

  @classmethod
  def apply(cls, dtm, bin_res, value='woe'):
    """
    Args:
      dtm:
      bin_res:
      value: 转换值类型，可选项('woe', 'index', 'bin')，默认woe.

    Returns:

    """
    special_values = bin_res['breaks'][bin_res['is_special_values']].tolist()
    if 'missing' in special_values:
      special_values.remove('missing')
    if len(special_values) == 0:
      special_values = None
    breaks = bin_res['breaks'][~bin_res['is_special_values']]
    dtm['value'] = replace_blank_string(dtm['value'])
    split_dtm = cls.split_special_values(dtm, special_values)
    dtm_sv = split_dtm['dtm_sv']
    dtm_ns = split_dtm['dtm_ns']

    if dtm_sv is not None:
      dtm_sv = dtm_sv[['idx', 'bin_chr', 'value', 'y']]

    if dtm_ns is not None:
      break_df = cls.split_vec_to_df(breaks)
      # TO-DO: 以下代码与binning_breaks重复，需要进行精简
      if is_numeric_dtype(dtm_ns['value']):
        break_list = ['-inf'] + list(
            set(break_df.value.tolist()).difference(
                {np.nan, '-inf', 'inf', 'Inf', '-Inf'})) + ['inf']
        break_list = sorted(list(map(float, break_list)))
        labels = [
            '[{},{})'.format(break_list[i], break_list[i + 1])
            for i in range(len(break_list) - 1)
        ]
        dtm_ns['bin_chr'] = pd.cut(
            dtm_ns['value'], break_list, right=False, labels=labels).astype(str)
      else:
        dtm_ns = pd.merge(dtm_ns, break_df, how='left', on='value')

      dtm_ns = dtm_ns[['idx', 'bin_chr', 'value', 'y']]

    new_dtm = pd.concat([dtm_sv, dtm_ns], ignore_index=True)
    dtm = pd.merge(dtm, new_dtm[['idx', 'bin_chr']], on='idx', how='left')
    bin_res = bin_res.copy()
    bin_res['index'] = bin_res.index
    bin_res['bin_chr'] = bin_res['bin']
    dtm = pd.merge(dtm, bin_res[['bin_chr', value]], on='bin_chr', how='left')
    dtm.set_index(dtm['idx'], drop=True, inplace=True)
    variable = dtm['variable'].iloc[0]
    feature_name = '_'.join([variable, value])
    dtm.rename(columns={value: feature_name}, inplace=True)
    return dtm[feature_name]

  def binning_breaks(self, dtm, breaks):
    """按照给定的breaks进行分箱"""
    break_df = self.split_vec_to_df(breaks)

    # binning
    if is_numeric_dtype(dtm['value']):
      break_list = ['-inf'] + list(
          set(break_df.value.tolist()).difference(
              {np.nan, '-inf', 'inf', 'Inf', '-Inf'})) + ['inf']
      break_list = sorted(list(map(float, break_list)))
      labels = [
          '[{},{})'.format(break_list[i], break_list[i + 1])
          for i in range(len(break_list) - 1)
      ]
      bin_chr = pd.cut(dtm['value'], break_list, right=False, labels=labels)

      binning = self.binning(dtm, bin_chr)

      # 不缺定经过正常流程是否还会产生空值，故注释掉下面代码，后续测试中如果发现确实
      # 存在空值再行处理。不建议在此处强行合并到missing中。
      # # sort bin
      # binning = pd.merge(
      #     binning.assign(value=lambda x: [
      #         float(re.search(r"^\[(.*),(.*)\)", i).group(2))
      #         if i != 'nan' else np.nan for i in binning['bin']
      #     ]),
      #     break_df.assign(value=lambda x: x.value.astype(float)),
      #     how='left',
      #     on='value').sort_values(by="rowid").reset_index(drop=True)
      # # merge binning and bk_df if nan isin value
      # if break_df['value'].isnull().any():
      #     binning = binning.assign(bin=lambda x:
      #     [i if i != 'nan' else 'missing' for i in x['bin']]) \
      #         .fillna('missing') \
      #         .groupby(['variable', 'rowid']) \
      #         .agg({'bin': lambda x: '%,%'.join(x),
      #               'good': sum,
      #               'bad': sum}) \
      #         .reset_index()
    else:
      dtm = pd.merge(dtm, break_df, how='left', on='value')
      binning = self.binning(dtm, dtm['bin_chr'])
      # 保持分箱顺序与传入参数一致
      binning['bin_chr'] = binning['bin_chr'].astype(
          'category').cat.set_categories(
              breaks, ordered=True)
      binning = binning.sort_values(by='bin_chr').reset_index(drop=True)

    return binning

  def binning_format(self, binning):
    """"""

    def sub0(x):
      """substitute 0"""
      return np.where(x == 0, self.epsilon, x)

    _pattern = re.compile(r"^\[(.*), *(.*)\)((%,%missing)*)")

    def _extract_breaks(x):
      gp23 = _pattern.match(x)
      breaks_string = x if gp23 is None else gp23.group(2)
      return breaks_string

    # yapf: disable
    binning = binning.assign(
      count=lambda x: x['good'] + x['bad'],
      bad_dist=lambda x: sub0(x['bad']) / sub0(x['bad']).sum(),
      good_dist=lambda x: sub0(x['good']) / sub0(x['good']).sum()
    ).assign(
      count_distr=lambda x: x['count'] / x['count'].sum(),
      badprob=lambda x: x['bad'] / x['count'],
      woe=lambda x: np.log(x['good_dist'] / x['bad_dist'])
    ).assign(
      bin_iv=lambda x: (x['good_dist'] - x['bad_dist']) * x['woe']
    ).assign(total_iv=lambda x: x['bin_iv'].sum())
    # yapf: enable

    binning['breaks'] = binning['bin_chr'].apply(_extract_breaks)
    binning['is_special_values'] = binning['is_sv']
    binning['bin'] = binning['bin_chr'].astype('str')

    return binning[[
        'variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob',
        'woe', 'bin_iv', 'total_iv', 'breaks', 'is_special_values'
    ]]


class ComposedWOEBin(WOEBin):
  """
    组合型分箱方法，将多个分箱方法按顺序进行调用。详细说明请见WOEBin。

    Args
        bin_object: 待组合WOEBin子类实例列表
    """

  def __init__(self, bin_objects, **kwargs):
    super().__init__(**kwargs)
    self.__bin_objects = bin_objects

  def woebin(self, dtm, breaks=None):
    for bin_obj in self.__bin_objects:
      breaks = bin_obj.woebin(dtm, breaks)

    return breaks


class InitBin(WOEBin):

  @staticmethod
  def check_empty_bins(dtm, breaks):
    """针对数值型变量检查空分箱"""
    bins = pd.cut(dtm['value'], breaks, right=False)
    bin_sample_count = bins.value_counts()
    if np.any(bin_sample_count == 0):
      bin_sample_count = bin_sample_count[bin_sample_count != 0]
      bin_right = set([
          re.match(r'\[(.+),(.+)\)', i).group(1)
          for i in bin_sample_count.index.astype('str')
      ]).difference({'-inf', 'inf'})
      breaks = sorted(list(map(float, ['-inf'] + list(bin_right) + ['inf'])))
    return breaks


@WOEBinFactory.register('quantile')
class QuantileInitBin(InitBin):
  """
    细分箱之等频分箱。对数值型变量，该分箱方法通过分位数寻找切分点。对类别型变量，直接返回所有
    类别值。

    Args:
        initial_bins: 等频分箱的箱数，默认20
        sig_figs: 切分点有效数字位数，默认4
    """

  def __init__(self, initial_bins=20, sig_figs=4, **kwargs):
    super().__init__(**kwargs)
    self.n_bins = initial_bins
    self.sig_figs = sig_figs

  def woebin(self, dtm, breaks=None):
    if is_numeric_dtype(dtm['value']):  # numeric variable
      xvalue = dtm['value'].astype(float)
      breaks = np.quantile(xvalue, np.linspace(0, 1, self.n_bins + 1))
      breaks = round_(np.unique(breaks), self.sig_figs)
      breaks[0] = -np.inf
      breaks[-1] = np.inf
      breaks = np.unique(breaks)
      breaks = self.check_empty_bins(dtm, breaks)
    else:
      breaks = np.unique(dtm['value'])
    return breaks


@WOEBinFactory.register('hist')
class HistogramInitBin(InitBin):
  """
    细分箱之等宽分箱。
    对类别型变量，直接返回所有类别值。
    对数值型变量，首先排除outlier样本，对剩余样本的range等分成`n_bins`等分。

    Args:
        n_bins: 等宽分箱的箱数。
    """

  @staticmethod
  def _pretty(low, high, n):

    def nice_number(x):
      exp = np.floor(np.log10(abs(x)))
      f = abs(x) / 10**exp
      if f < 1.5:
        nf = 1.
      elif f < 3.:
        nf = 2.
      elif f < 7.:
        nf = 5.
      else:
        nf = 10.
      return np.sign(x) * nf * 10.**exp

    d = abs(nice_number((high - low) / (n - 1)))
    min_x = np.floor(low / d) * d
    max_x = np.ceil(high / d) * d
    return np.arange(min_x, max_x + 0.5 * d, d)

  # @staticmethod
  # def _check_empty_bins(dtm, breaks):
  #     """针对数值型变量检查空分箱"""
  #     bins = pd.cut(dtm['value'], breaks, right=False)
  #     bin_sample_count = bins.value_counts()
  #     if np.any(bin_sample_count == 0):
  #         bin_sample_count = bin_sample_count[bin_sample_count != 0]
  #         bin_right = set([
  #             re.match(r'\[(.+),(.+)\)', i).group(1)
  #             for i in bin_sample_count.index.astype('str')
  #         ]).difference({'-inf', 'inf'})
  #         breaks = sorted(
  #             list(map(float, ['-inf'] + list(bin_right) + ['inf'])))
  #     return breaks

  def woebin(self, dtm, breaks=None):
    if is_numeric_dtype(dtm['value']):  # numeric variable
      xvalue = dtm['value'].astype(float)

      # outlier处理
      iq = xvalue.quantile([0.01, 0.25, 0.75, 0.99])
      iqr = iq[0.75] - iq[0.25]
      if iqr == 0:
        prob_down = 0.01
        prob_up = 0.99
      else:
        prob_down = 0.25
        prob_up = 0.75
      xvalue_rm_outlier = xvalue[(xvalue >= iq[prob_down] - 3 * iqr)
                                 & (xvalue <= iq[prob_up] + 3 * iqr)]

      n_bins = self.n_bins
      len_uniq_x = len(np.unique(xvalue_rm_outlier))
      if len_uniq_x < n_bins:
        n_bins = len_uniq_x
      # initial breaks
      if len_uniq_x == n_bins:
        breaks = np.unique(xvalue_rm_outlier)
      else:
        breaks = self._pretty(
            low=min(xvalue_rm_outlier),
            high=max(xvalue_rm_outlier),
            n=self.n_bins)

      breaks = list(
          filter(lambda x: np.nanmin(xvalue) < x <= np.nanmax(xvalue), breaks))
      breaks = [float('-inf')] + sorted(breaks) + [float('inf')]
      breaks = self.check_empty_bins(dtm, breaks)
    else:
      breaks = np.unique(dtm['value'])
    return breaks

  def __init__(self, initial_bins=20, **kwargs):
    super().__init__(**kwargs)
    self.n_bins = initial_bins


class OptimBinMixin:
  """粗分箱Mixin，提供initial_binning方法，根据细分箱切分点生成分享统计表"""

  def initial_binning(self, dtm, breaks):
    binning = self.binning_breaks(dtm, breaks)
    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count'] / binning['count'].sum()

    if not is_numeric_dtype(dtm['value']):
      binning['badprob'] = binning['bad'] / binning['count']
      binning = binning.sort_values(
          by='badprob', ascending=False).reset_index(drop=True)

    return binning


@WOEBinFactory.register(['chi2', 'chimerge'])
class ChiMergeOptimBin(WOEBin, OptimBinMixin):
  """
    ChiMerge最优分箱方法，对相邻分箱进行chi2列联表独立性检验，基于检验的统计量进行分箱合并。

    TODO: 增加分箱单调性约束

    Args
        - bin_num_limit: 分箱数上限，默认5
        - p: 独立性检验显著性，模型0.05
        - count_distr_limit: 最小分箱样本占比，默认0.02
        - ensure_monotonic: 是否要求单调，默认False（暂不支持该功能）
    """

  def __init__(self,
               bin_num_limit=5,
               p=0.05,
               count_distr_limit=0.02,
               ensure_monotonic=False,
               **kwargs):
    super().__init__(**kwargs)
    self.bin_num_limit = bin_num_limit
    self.p = p
    self.count_distr_limit = count_distr_limit
    self.ensure_monotonic = ensure_monotonic
    self.chi2_limit = chi2.isf(p, df=1)

  @staticmethod
  def chi2_stat(binning):
    """计算两分箱之间的Chi2统计量，这里直接使用`scipy.stats.chi2_contingency`函数，
        并且使用 Yate's 连续性修正"""
    binning['good_lag'] = binning['good'].shift(1)
    binning['bad_lag'] = binning['bad'].shift(1)

    def chi2_cont_tbl(arr):
      if np.any(np.isnan(arr)):
        return np.nan
      elif np.any(np.sum(arr, axis=1) == 0) or np.any(np.sum(arr, axis=0) == 0):
        return 0.0
      else:
        return chi2_contingency(arr, correction=True)[0]

    binning['chi2'] = binning.apply(
        lambda x: chi2_cont_tbl([[x['good'], x['bad']],
                                 [x['good_lag'], x['bad_lag']]]),
        axis=1)
    del binning['good_lag']
    del binning['bad_lag']

    return binning

  def woebin(self, dtm, breaks=None):
    assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，" \
                               f"需要传入初始分箱（细分箱）结果"
    binning = self.initial_binning(dtm, breaks)
    binning_chi2 = self.chi2_stat(binning)
    binning_chi2['bin_chr'] = binning_chi2['bin_chr'].astype('str')

    # Start merge loop
    while True:
      min_chi2 = binning_chi2['chi2'].min()
      min_count_distr = binning_chi2['count_distr'].min()
      n_bins = len(binning_chi2)

      if min_chi2 < self.chi2_limit:
        # 分箱坏占比差异不显著
        idx = binning_chi2[binning_chi2['chi2'] == min_chi2].index[0]
      elif min_count_distr < self.count_distr_limit:
        # 分箱占比过少
        idx = binning_chi2[binning_chi2['count_distr'] ==
                           min_count_distr].index[0]
        if idx == 0 or (idx < len(binning_chi2) - 1 and
                        (binning_chi2['chi2'][idx]
                         > binning_chi2['chi2'][idx + 1])):
          idx = idx + 1
      elif n_bins > self.bin_num_limit:
        # 分箱数太多
        idx = binning_chi2[binning_chi2['chi2'] == min_chi2].index[0]
      else:
        # 结束合并操作
        break

      # yapf: disable
      binning_chi2.loc[idx - 1, 'bin_chr'] = '%,%'.join(
        [binning_chi2.loc[idx - 1, 'bin_chr'],
         binning_chi2.loc[idx, 'bin_chr']])
      binning_chi2.loc[idx - 1, 'count'] = (
          binning_chi2.loc[idx - 1, 'count'] +
          binning_chi2.loc[idx, 'count'])
      binning_chi2.loc[idx - 1, 'count_distr'] = (
          binning_chi2.loc[idx - 1, 'count_distr'] +
          binning_chi2.loc[idx, 'count_distr'])
      binning_chi2.loc[idx - 1, 'good'] = (
          binning_chi2.loc[idx - 1, 'good'] +
          binning_chi2.loc[idx, 'good'])
      binning_chi2.loc[idx - 1, 'bad'] = (
          binning_chi2.loc[idx - 1, 'bad'] +
          binning_chi2.loc[idx, 'bad'])
      # yapf: enable

      if is_numeric_dtype(dtm['value']):
        # 数值类型分箱合并
        # [a,b)%,%[b,c) → [a,c)
        binning_chi2['bin_chr'] = binning_chi2['bin_chr'].apply(
            lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))

      index = binning_chi2.index.tolist()
      index.remove(idx)
      binning_chi2 = binning_chi2.iloc[
          index,
      ].reset_index(drop=True)
      binning_chi2 = self.chi2_stat(binning_chi2)
    # End of loop

    # 切分点提取
    if is_numeric_dtype(dtm['value']):
      _pattern = re.compile(r"^\[(.*), *(.*)\)")
      breaks = binning_chi2['bin_chr'].apply(lambda x: _pattern.match(x)[2])
      breaks = pd.to_numeric(breaks)
    else:
      breaks = binning_chi2['bin_chr']

    return breaks


@WOEBinFactory.register('tree')
class TreeOptimBin(WOEBin, OptimBinMixin):
  """
    树分箱方法，从细分箱生成的切分点中挑选最优切分点，自顶向下逐步生成分箱树，完成分箱。

    Args
        bin_num_limit: 分箱数上限，默认5
        min_iv_inc: 增加切分点后IV相对增幅最小值，模型0.05
        count_distr_limit: 最小分箱样本占比，默认0.02
        ensure_monotonic: 是否要求严格单调，默认False
    """

  def __init__(self,
               bin_num_limit=5,
               min_iv_inc=0.05,
               count_distr_limit=0.02,
               ensure_monotonic=False,
               **kwargs):
    super().__init__(**kwargs)
    self.bin_num_limit = bin_num_limit
    self.min_iv_inc = min_iv_inc
    self.count_distr_limit = count_distr_limit
    self.ensure_monotonic = ensure_monotonic

  def woebin(self, dtm, breaks=None):
    assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，" \
                               f"需要传入初始分箱（细分箱）结果"
    binning_tree = self.initial_binning(dtm, breaks)
    binning_tree['node_id'] = 0
    binning_tree['cp'] = False
    binning_tree.loc[len(binning_tree) - 1, 'cp'] = True

    last_iv = 0

    while len(binning_tree['node_id'].unique()) <= self.bin_num_limit:
      cut_idx_iv = {}
      for idx in binning_tree.index[~binning_tree['cp']]:
        new_node_ids = self.node_split(binning_tree['node_id'], idx)
        new_binning = self.merge_binning(binning_tree, new_node_ids)
        if self.ensure_monotonic:
          monotonic_type = monotonic(new_binning['bad_prob'])
          if monotonic_type in ('increasing', 'decreasing'):
            monotonic_constrain = True
          else:
            monotonic_constrain = False
        else:
          monotonic_constrain = True

        if (np.all(new_binning['count_distr'] > self.count_distr_limit) and
            monotonic_constrain):
          # (pandas==2.1.1)
          # FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version,
          # integer keys will always be treated as labels (consistent with DataFrame behavior). To access a
          # value by position, use `ser.iloc[pos]`
          # curr_iv = new_binning['total_iv'][0]
          curr_iv = new_binning['total_iv'].iloc[0]
          if ((curr_iv - last_iv + 1e-8) / (last_iv + 1e-8)) > self.min_iv_inc:
            cut_idx_iv[idx] = curr_iv

      if len(cut_idx_iv) > 0:
        sorted_cut_idx_iv = sorted(cut_idx_iv.items(), key=lambda x: -x[1])
        best_cut_idx = sorted_cut_idx_iv[0][0]
        last_iv = sorted_cut_idx_iv[0][1]
        binning_tree['node_id'] = self.node_split(binning_tree['node_id'],
                                                  best_cut_idx)
        binning_tree.loc[best_cut_idx, 'cp'] = True
      else:
        break

    best_binning = self.merge_binning(binning_tree, binning_tree['node_id'])

    if is_numeric_dtype(dtm['value']):
      best_binning['bin_chr'] = best_binning['bin_chr'].apply(
          lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))
      _pattern = re.compile(r"^\[(.*), *(.*)\)")
      breaks = best_binning['bin_chr'].apply(lambda x: _pattern.match(x)[2])
      breaks = pd.to_numeric(breaks)
    else:
      breaks = best_binning['bin_chr']

    return breaks

  def merge_binning(self, binning, node_ids):
    # yapf: disable
    new_binning = binning.groupby([
      'variable',
      node_ids,
    ]).agg(bin_chr=('bin_chr', lambda x: '%,%'.join(x.tolist())),
           count=('count', 'sum'),
           count_distr=('count_distr', 'sum'),
           good=('good', 'sum'),
           bad=('bad', 'sum')
           ).assign(
      bad_prob=lambda x: x['bad'] / x['count'],
      total_iv=lambda x: self.iv(x['good'], x['bad']))
    # yapf: enable

    return new_binning

  @staticmethod
  def node_split(node_ids, idx):
    new_node_ids = np.where(node_ids.index <= idx, node_ids, node_ids + 1)

    return new_node_ids

  def iv(self, good, bad):
    good = np.asarray(good)
    bad = np.asarray(bad)
    # substitute 0 by self.epsilon
    good = np.where(good == 0, self.epsilon, good)
    bad = np.where(bad == 0, self.epsilon, bad)
    good_distr = good / good.sum()
    bad_distr = bad / bad.sum()
    iv = (good_distr - bad_distr) * np.log(good_distr / bad_distr)
    return iv.sum()


@nb.njit
def foil():
  ...


class RuleOptimBin(WOEBin, OptimBinMixin):
  """规则优化分箱算法，用于生成单变量规则。

   required_list: 要求最小风险提升度，默认为 3
   min_hit_samples: 最小命中样本数，默认为 None 代表不限制命中样本数
   """

  def __init__(self,
               required_lift: float = 3,
               min_hit_samples: Optional[int] = None):
    super().__init__()
    self._required_lift = required_lift
    self._min_hit_samples = min_hit_samples

  def woebin(self, dtm, breaks=None):
    assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，" \
                               f"需要传入初始分箱（细分箱）结果"
    binning = self.initial_binning(dtm, breaks)
    ...


def woebin_ply(dt, bins, no_cores=None, replace_blank=False, value='woe'):
  """
    将woebin函数返回分箱结果进行应用，与`scorecardpy.woebin_ply`相比，增加参数`value`。
    该参数可选值为['woe', 'index', 'bin']，当 value='woe' 时，将原始值替换为woe值；
    当 value='index' 时，将原始值替换为变量分箱结果数据框中的 index ，即0, 1, 2,...，可用于
    one-hot编码等进一步处理；当 value='bin' 时，返回结果为分箱区间 [a,b) 【数值型变量】或
    a%,%b 【类别型变量】。

    注意，对应不同 value 参数值，返回数据框中变量名后缀会相应改变。假设原始变量名为 'var_123'，
    返回数据框中列名对应为 'var_123_{value}'。

    Args:
        dt: 包含变量原始值的数据框
        bins: woebin分箱结果
        no_cores: 多进程数量
        replace_blank: 是否将空字符串 '' 替换为 np.nan。注意若为 True 本函数非常耗时，
            建议在预处理阶段就完成该操作。保留该参数是为了保持与 scorecardpy 的兼容性。
        value: 返回值类别，可选['woe', 'index', 'bin']。

    Returns:
        pd.DataFrame，包含入参数据框中未替换的所有列，和替换后的变量列。

    """
  # start time
  start_time = time.time()

  # x variables
  x_vars_bin = bins.keys()
  x_vars_dt = dt.columns.tolist()
  x_vars = list(set(x_vars_bin).intersection(x_vars_dt))
  n_x = len(x_vars)
  # initial data set
  dat = dt.loc[:, list(set(x_vars_dt) - set(x_vars))].copy()

  if no_cores is None or no_cores < 1:
    all_cores = mp.cpu_count() - 1
    no_cores = int(np.ceil(n_x / 5 if n_x / 5 < all_cores else all_cores * 0.9))
  no_cores = max(no_cores, 1)

  tasks = [
      (
          pd.DataFrame({
              'y': 0,  # 不重要
              'variable': var,
              'value': dt[var]
          }),
          bins[var],
          value) for var in x_vars
  ]

  if no_cores == 1:
    dat_suffix = list(itertools.starmap(WOEBin.apply, tasks))
  else:
    pool = mp.Pool(processes=no_cores)
    dat_suffix = pool.starmap(WOEBin.apply, tasks)
    pool.close()

  dat = pd.concat([dat] + dat_suffix, axis=1)

  # running time
  running_time = time.time() - start_time
  logging.info('Woe transformation on {} rows and {} columns in {}'.format(
      dt.shape[0], n_x, time.strftime("%H:%M:%S", time.gmtime(running_time))))
  return dat


def plot_bin(binx, title, show_iv):
  y_right_max = np.ceil(binx['badprob'].max() * 10)
  if y_right_max % 2 == 1:
    y_right_max = y_right_max + 1
  if y_right_max - binx['badprob'].max() * 10 <= 0.3:
    y_right_max = y_right_max + 2
  y_right_max = y_right_max / 10
  if (y_right_max > 1 or y_right_max <= 0 or y_right_max is np.nan or
      y_right_max is None):
    y_right_max = 1
  # y_left_max
  y_left_max = np.ceil(binx['count_distr'].max() * 10) / 10
  if (y_left_max > 1 or y_left_max <= 0 or y_left_max is np.nan or
      y_left_max is None):
    y_left_max = 1
  # title
  title_string = binx.loc[0, 'variable'] + "  (iv:" + str(
      round(binx.loc[0,
                     'total_iv'], 4)) + ")" if show_iv else binx.loc[0,
                                                                     'variable']
  title_string = (
      title + '-' + title_string if title is not None else title_string)
  # param
  ind = np.arange(len(binx.index))  # the x locations for the groups
  width = 0.35  # the width of the bars: can also be len(x) sequence
  # plot
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  # ax1
  p1 = ax1.bar(
      ind, binx['good_distr'], width, color=(24 / 254, 192 / 254, 196 / 254))
  p2 = ax1.bar(
      ind,
      binx['bad_distr'],
      width,
      bottom=binx['good_distr'],
      color=(246 / 254, 115 / 254, 109 / 254))
  for i in ind:
    ax1.text(
        i,
        binx.loc[i, 'count_distr'] * 1.02,
        str(round(binx.loc[i, 'count_distr'] * 100, 1)) + '%, ' +
        str(binx.loc[i, 'count']),
        ha='center')
  # ax2
  ax2.plot(ind, binx['badprob'], marker='o', color='blue')
  for i in ind:
    ax2.text(
        i,
        binx.loc[i, 'badprob'] * 1.02,
        str(round(binx.loc[i, 'badprob'] * 100, 1)) + '%',
        color='blue',
        ha='center')
  # settings
  ax1.set_ylabel('Bin count distribution')
  ax2.set_ylabel('Bad probability', color='blue')
  ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
  ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
  ax2.tick_params(axis='y', colors='blue')
  plt.xticks(ind, binx['bin'])
  plt.title(title_string, loc='left')
  plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')

  return fig


def woebin_plot(bins, x=None, title=None, show_iv=True):
  xs = x
  # bins concat
  if isinstance(bins, dict):
    bins = pd.concat(bins, ignore_index=True)

  # good bad distr
  def gb_distr(bin_x):
    bin_x['good_distr'] = bin_x['good'] / sum(bin_x['count'])
    bin_x['bad_distr'] = bin_x['bad'] / sum(bin_x['count'])
    return bin_x

  bins = bins.groupby('variable').apply(gb_distr)
  # x variable names
  if xs is None:
    xs = bins['variable'].unique()
  # plot export
  plot_list = {}
  for i in xs:
    binx = bins[bins['variable'] == i].reset_index()
    plot_list[i] = plot_bin(binx, title, show_iv)
  return plot_list


def sc_bins_to_df(sc_bins):
  """
    将 woebin 返回的结果转换为 woe 数据框、iv 数据框
    Args:
        sc_bins: dict, 由 woebin 返回，结构为 {'VAR_NAME': 'BIN_STATS'}

    Returns:
        (woe_df, iv_df)

    """
  woe_df = None
  for key, value in sc_bins.items():
    if isinstance(value, pd.DataFrame):
      if woe_df is None:
        woe_df = value
      else:
        woe_df = pd.concat([woe_df, value], axis=0, ignore_index=True)

  def iv_stats(x):
    iv = x.total_iv.max()
    badrate = x['bad'].sum() / x['count'].sum()
    lift = x.badprob / badrate
    iv_interval = None
    if iv < 0.02:
      iv_interval = '(0, 0.02)'
    elif iv < 0.05:
      iv_interval = '[0.02, 0.05)'
    elif iv < 0.08:
      iv_interval = '[0.05, 0.08)'
    elif iv < 0.1:
      iv_interval = '[0.08, 0.1)'
    elif iv < 0.2:
      iv_interval = '[0.1, 0.2)'
    else:
      iv_interval = '[0.2, +)'

    badrate = x[~x.is_special_values].badprob
    monotonic_type = monotonic(badrate)

    return pd.Series(
        [iv, iv_interval, monotonic_type,
         lift.max(), lift.min()],
        index=['IV', 'IV区间', '单调性', '最大Lift', '最小Lift'],
        dtype='object')

  if woe_df is None:
    return None, None
  else:
    iv_df = woe_df.groupby(by='variable').apply(iv_stats)
    iv_df.sort_values(by='IV', ascending=False, inplace=True)
    return woe_df, iv_df


def make_scorecard(sc_bins, coef, *, base_points=600, base_odds=50, pdo=20):
  a = pdo / np.log(2)
  b = base_points - a * np.log(base_odds)

  base_score = -a * coef['const'] + b
  score_df = [
      pd.DataFrame({
          'variable': ['base score'],
          'bin': [''],
          'woe': [''],
          'score': [base_score]
      })
  ]

  for var in coef.keys():
    if var != 'const':
      woe_df = sc_bins[var[:-4]][['variable', 'bin', 'woe']].copy()
      woe_df['score'] = -a * coef[var] * woe_df['woe']
      score_df.append(woe_df)

  score_df = pd.concat(score_df, ignore_index=True)
  score_df['score'] = np.round(score_df['score'], 2)
  return score_df


def woebin_breaks(bins):
  """
    从woebin返回结果中提取切分点及特殊值
    Args:
        bins: woebin函数的返回结果

    Returns:
        breaks字典和special_values字典组成的元组

    """

  def get_breaks(binning):
    if np.any(binning['is_special_values']):
      special_values = binning[binning['is_special_values']]['breaks']
      special_values = special_values.tolist()

      # 注意，此处需要将 missing 从 special_values 中排除。否则当 dtm 无空值
      # 且 value 为数值型，而 special_values 中存在 missing 时，
      # WOEBin.split_special_values 中 merge 步会报如下错误：
      # ValueError: You are trying to merge on float64 and object columns.
      # If you wish to proceed you should use pd.concat

      if 'missing' in special_values:
        special_values.remove('missing')
      if len(special_values) == 0:
        special_values = None

    else:
      special_values = None

    breaks = binning[~binning['is_special_values']]['breaks']
    breaks = breaks.tolist()
    return {'breaks': breaks, 'special_values': special_values}

  brk_spcs = {key: get_breaks(value) for key, value in bins.items()}
  breaks = {k: v['breaks'] for k, v in brk_spcs.items()}
  special_values = {
      k: v['special_values'] for k, v in brk_spcs.items() if v['special_values']
  }
  return breaks, special_values


def woebin_psi(df_base, df_cmp, bins):
  """
    计算变量PSI

    Args:
        df_base: 基准数据集，一般为训练集
        df_cmp: 比较数据集，一般为测试集、OOT等等
        bins: 变量分箱结果，由`woebin`返回

    Returns:
        pd.DataFrame，包含variable, bin, base%, cmp%, total_psi五列

    """

  # replace_blank is very expensive, so set the arg to False. But this may
  # cause errors. A good practice is replacing the blank str to np.nan in
  # data preprocessing procedure.
  X_base = woebin_ply(df_base, bins, value='bin', replace_blank=False)
  X_cmp = woebin_ply(df_cmp, bins, value='bin', replace_blank=False)

  vars_base = [v for v in X_base.columns if v.endswith('_bin')]
  vars_cmp = [v for v in X_cmp.columns if v.endswith('_bin')]
  variables = list(set(vars_base).intersection(set(vars_cmp)))

  X_base['set'] = 'base'
  X_cmp['set'] = 'cmp'

  dat = pd.concat([X_base, X_cmp])
  dat['idx'] = dat.index

  psi_dfs = []

  for variable in variables:
    psi_df = pd.pivot_table(
        dat, index=variable, columns=['set'], values=['idx'], aggfunc='count')
    psi_df.columns = ['base', 'cmp']
    psi_df['variable'] = variable[:-4]
    psi_df['bin'] = psi_df.index
    psi_df.reset_index(drop=True, inplace=True)

    psi_df = psi_df.assign(
        base_distr=lambda x: x['base'] / x['base'].sum(),
        cmp_distr=lambda x: x['cmp'] / x['cmp'].sum()).assign(
            psi=lambda x: psi(x['base_distr'], x['cmp_distr']))[[
                'variable', 'bin', 'base_distr', 'cmp_distr', 'psi'
            ]]
    psi_dfs.append(psi_df)

  return pd.concat(psi_dfs, ignore_index=True)
