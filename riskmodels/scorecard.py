# -*- encoding: utf-8 -*-
import itertools
import multiprocessing as mp
import platform
import re
import time
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

__all__ = ['interactive_mode', 'woebin', 'WOEBin']


def interactive_mode():
    # noinspection PyPackageRequirements
    import __main__ as main
    if hasattr(main, '__file__'):
        return False
    else:
        return True


def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x


def check_const_cols(dat):
    # remove only 1 unique values variable
    unique1_cols = [i for i in list(dat) if len(dat[i].unique()) == 1]
    if len(unique1_cols) > 0:
        warnings.warn(
            "There are {} columns have only one unique values, which are "
            "removed from input dataset. \n (ColumnNames: {})".format(
                len(unique1_cols), ', '.join(unique1_cols)))
        dat = dat.drop(unique1_cols, axis=1)
    return dat


def check_datetime_cols(dat):
    datetime_cols = dat.apply(
        pd.to_numeric, errors='ignore').select_dtypes(object).apply(
            pd.to_datetime,
            errors='ignore').select_dtypes('datetime64').columns.tolist()
    if len(datetime_cols) > 0:
        warnings.warn(
            "There are {} date/time type columns are removed from input "
            "dataset. \n (ColumnNames: {})".format(len(datetime_cols),
                                                   ', '.join(datetime_cols)))
        dat = dat.drop(datetime_cols, axis=1)
    return dat


def check_cat_var_uniques(dat, unique_limits=50, var_skip=None):
    # character columns with too many unique values
    char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
    if var_skip is not None:
        var_skip = str_to_list(var_skip)
        char_cols = list(set(char_cols) - set(str_to_list(var_skip)))
    cat_var_too_many_unique = [
        i for i in char_cols if len(dat[i].unique()) >= unique_limits
    ]
    if len(cat_var_too_many_unique) > 0:
        print(
            f'>>> There are {cat_var_too_many_unique} categorical variables '
            f'with too many unique values (>= {unique_limits}). Please double '
            'check the following  variables: \n'
            f'{", ".join(cat_var_too_many_unique)}')
        print('>>> Skip these variables?')
        print('1: yes \n2: no')
        cont = int(input("Selection: "))
        while cont not in [1, 2]:
            cont = int(input("Selection: "))
        if cont == 1:
            if var_skip is None:
                var_skip = cat_var_too_many_unique
            else:
                var_skip.extend(cat_var_too_many_unique)
    return var_skip


def rep_blank_na(dat):
    # cant replace blank string in categorical value with nan
    # remove duplicated index
    if dat.index.duplicated().any():
        dat = dat.reset_index(drop=True)
        warnings.warn(
            'There are duplicated index in dataset. The index has been reset.')

    blank_cols = [
        i for i in list(dat) if dat[i].astype(str).str.findall(r'^\s*$').apply(
            lambda x: 0 if len(x) == 0 else 1).sum() > 0
    ]
    if len(blank_cols) > 0:
        warnings.warn(
            'There are blank strings in {} columns, which are replaced with '
            'NaN. \n (ColumnNames: {})'.format(len(blank_cols),
                                               ', '.join(blank_cols)))

        dat.replace(r'^\s*$', np.nan, regex=True)

    return dat


def check_y(dat, y, positive):
    positive = str(positive)
    if not isinstance(dat, pd.DataFrame):
        raise Exception("Incorrect inputs; dat should be a DataFrame.")
    elif dat.shape[1] <= 1:
        raise Exception(
            "Incorrect inputs; dat should be a DataFrame with at least "
            "two columns.")

    y = str_to_list(y)
    if len(y) != 1:
        raise Exception("Incorrect inputs; the length of y should be one")

    y = y[0]
    # y not in dat.columns
    if y not in dat.columns:
        raise Exception(
            "Incorrect inputs; there is no \'{}\' column in dat.".format(y))

    # remove na in y
    if dat[y].isnull().any():
        warnings.warn(
            "There are NaNs in \'{}\' column. The rows with NaN in \'{}\' were"
            " removed from dat.".format(y, y))
        dat = dat.dropna(subset=[y])

    # numeric y to int
    if is_numeric_dtype(dat[y]):
        dat.loc[:, y] = dat[y].apply(lambda x: x if pd.isnull(x) else int(x))
    # length of unique values in y
    unique_y = np.unique(dat[y].values)
    if len(unique_y) == 2:
        # if [v not in [0,1] for v in unique_y] == [True, True]:
        if True in [bool(re.search(positive, str(v))) for v in unique_y]:
            y1 = dat[y]
            y2 = dat[y].apply(lambda x: 1
                              if str(x) in re.split(r'\|', positive) else 0)
            if np.any(y1 != y2):
                dat.loc[:, y] = y2
                warnings.warn(
                    "The positive value in \"{}\" was replaced by 1 and "
                    "negative value by 0.".format(y))
        else:
            raise Exception("Incorrect inputs; the positive value in \"{}\" is "
                            "not specified".format(y))
    else:
        raise Exception("Incorrect inputs; the length of unique values in y "
                        "column \'{}\' != 2.".format(y))

    return dat


def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step < 0:
        warnings.warn(
            "Incorrect inputs; print_step should be a non-negative integer. "
            "It was set to 1.")
        print_step = 1
    return print_step


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
                warnings.warn(
                    "Incorrect inputs; there are {} x variables are not exist "
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
            warnings.warn(
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


def woebin(dt,
           y,
           x=None,
           var_skip=None,
           breaks_list=None,
           special_values=None,
           stop_limit=0.1,
           init_count_distr=0.02,
           count_distr_limit=0.05,
           bin_num_limit=8,
           positive="bad|1",
           no_cores=None,
           print_step=0,
           method="tree",
           ignore_const_cols=True,
           ignore_datetime_cols=True,
           check_cate_num=True,
           replace_blank=True,
           save_breaks_list=None,
           **kwargs):

    # start time
    start_time = time.time()

    # arguments
    print_info = kwargs.get('print_info', True)
    print_step = check_print_step(print_step)
    # stop_limit range
    if stop_limit < 0 or stop_limit > 0.5 or not isinstance(
            stop_limit, (float, int)):
        warnings.warn(
            "Incorrect parameter specification; accepted stop_limit parameter "
            "range is 0-0.5. Parameter was set to default (0.1).")
        stop_limit = 0.1
    # init_count_distr range
    if init_count_distr < 0.01 or init_count_distr > 0.2 or not isinstance(
            init_count_distr, (float, int)):
        warnings.warn(
            "Incorrect parameter specification; accepted init_count_distr "
            "parameter range is 0.01-0.2. Parameter was set to default (0.02).")
        init_count_distr = 0.02
    # count_distr_limit
    if count_distr_limit < 0.01 or count_distr_limit > 0.2 or not isinstance(
            count_distr_limit, (float, int)):
        warnings.warn(
            "Incorrect parameter specification; accepted count_distr_limit "
            "parameter range is 0.01-0.2. Parameter was set to default (0.05).")
        count_distr_limit = 0.05
    # bin_num_limit
    if not isinstance(bin_num_limit, (float, int)):
        warnings.warn(
            "Incorrect inputs; bin_num_limit should be numeric variable. "
            "Parameter was set to default (8).")
        bin_num_limit = 8
    # method
    if method not in ["tree", "chimerge"]:
        warnings.warn("Incorrect inputs; method should be tree or chimerge. "
                      "Parameter was set to default (tree).")
        method = "tree"

    # print information
    if print_info:
        print('[INFO] creating woe binning ...')

    dt = dt.copy(deep=True)
    y = str_to_list(y)
    x = str_to_list(x)
    if x is not None:
        dt = dt[y + x]
    # check y
    dt = check_y(dt, y, positive)
    # remove constant columns
    if ignore_const_cols:
        dt = check_const_cols(dt)
    # remove date/time col
    if ignore_datetime_cols:
        dt = check_datetime_cols(dt)
    # check categorical columns' unique values
    if check_cate_num:
        var_skip = check_cat_var_uniques(dt, var_skip)
    # replace black with na
    if replace_blank:
        dt = rep_blank_na(dt)

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
    if platform.system() == 'Windows' and interactive_mode():
        no_cores = 1

    # y list to str
    y = y[0]
    # tasks for binning variable

    woe_bin = woebin_factory(init_count_distr, count_distr_limit, stop_limit,
                             bin_num_limit, method)

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
            special_values.get(x_i)) for x_i in xs
    ]

    if no_cores == 1:
        bins = dict(zip(xs, itertools.starmap(woe_bin, tasks)))
    else:
        pool = mp.Pool(processes=no_cores)
        bins = dict(zip(xs, pool.starmap(woe_bin, tasks)))
        pool.close()

    # running time
    running_time = time.time() - start_time
    if print_info:
        print('Binning on {} rows and {} columns in {}'.format(
            dt.shape[0], len(xs),
            time.strftime("%H:%M:%S", time.gmtime(running_time))))
    # if save_breaks_list is not None:
    #     bins_to_breaks(bins, dt, to_string=True, save_string=save_breaks_list)
    return bins


class WOEBin(object):
    """WOEBin: 对单个变量进行分箱操作的基类，所有分箱操作的类都继承本类。

    分箱类型包括细分箱、粗分箱。其中细分箱用来初始化分箱，一般是通过等频(quantile)或等宽
    (histogram)的方式进行分箱；粗分箱在粗分箱结果的基础上进行，用来简化分箱结果，确保稳定
    性和显著性。

    * 常用变量约定

    dtm定义:
        下述所有dtm变量为（至少）包含 variable（变量名）、y（目标变量）、value（解释
        变量值）三列的数据框（pd.DataFrame）。其中value可包含空值和其他特殊值

    ns_dtm：不含空值、特殊值的dtm（ns for non-special）

    binning_*：
        分箱信息统计表，包含 variable（变量名）、bin（分箱名）、good（好样本数）、
        bad（坏样本数）四列的数据框（pd.DataFrame）。

    Args
        eps: 若某一分箱中好样本或坏样本数为0，计算woe及iv时用eps替换0，默认值0.5
    """

    def __init__(self, eps=0.5):
        self.epsilon = eps

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
        特殊值列表转DataFrame, 如下例所示：

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
        if vec is not None:
            vec = [str(i) for i in vec]
            a = pd.DataFrame({'bin_chr': vec}).assign(rowid=lambda x: x.index)
            b = pd.DataFrame([i.split('%,%') for i in vec], index=vec) \
                .stack().replace('missing', np.nan) \
                .reset_index(name='value') \
                .rename(columns={'level_0': 'bin_chr'})[['bin_chr', 'value']]

            return pd.merge(a, b, on='bin_chr')

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
        spl_val = cls.add_missing_spl_val(dtm, spl_val)
        if spl_val is not None:
            sv_df = cls.split_vec_to_df(spl_val)
            # value
            if is_numeric_dtype(dtm['value']):
                sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
                sv_df['bin_chr'] = np.where(np.isnan(sv_df['value']),
                                            sv_df['bin_chr'],
                                            sv_df['value'].astype(str))
            # dtm_sv & dtm
            dtm_merge = pd.merge(dtm.fillna("missing"),
                                 sv_df[['value', 'rowid']].fillna("missing"),
                                 how='left',
                                 on='value')
            dtm_sv = dtm_merge[~dtm_merge['rowid'].isna()][
                dtm.columns.tolist()].reset_index(drop=True)
            ns_dtm = dtm_merge[dtm_merge['rowid'].isna()][
                dtm.columns.tolist()].reset_index(drop=True)
            if len(ns_dtm) == 0:
                ns_dtm = None

            if dtm_sv.shape[0] == 0:
                binning_sv = None
            else:
                dtm_sv = pd.merge(dtm_sv.fillna('missing'),
                                  sv_df.fillna('missing'),
                                  on='value')
                binning_sv = cls.binning(dtm, dtm_sv['bin_chr'])
        else:
            binning_sv = None
            ns_dtm = dtm

        return {'binning_sv': binning_sv, 'ns_dtm': ns_dtm}

    def __call__(self, dtm, breaks, special_values):
        binning_split = self.dtm_binning_sv(dtm, special_values)
        binning_sv = binning_split['binning_sv']
        dtm = binning_split['ns_dtm']

        if dtm is None:
            binning_dtm = None
        else:
            if breaks is not None:
                binning_dtm = self._binning_breaks(dtm, breaks)
            else:
                breaks = self.woebin(dtm)
                binning_dtm = self._binning_breaks(dtm, breaks)

        bin_list = {'binning_sv': binning_sv, 'binning': binning_dtm}
        binning = pd.concat(bin_list, keys=bin_list.keys()).reset_index() \
            .assign(is_sv=lambda x: x.level_0 == 'binning_sv')

        return self.binning_format(binning)

    def woebin(self, dtm, breaks=None):
        """"""
        raise NotImplementedError

    @classmethod
    def binning(cls, dtm, bin_chr):
        """
        给定dtm、分箱序列生成binning，dtm和binning的定义见WOEBin类文档。bin_chr为
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
        binning = dtm.groupby(['variable', bin_chr])['y'].agg(good=_n0, bad=_n1)
        binning = binning.reset_index()

        return binning

    def _binning_breaks(self, dtm, breaks):
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
            bin_chr = pd.cut(dtm['value'],
                             break_list,
                             right=False,
                             labels=labels).astype(str)

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
        return binning

    def binning_format(self, binning):
        """"""

        def _rm0(x):
            return np.where(x == 0, self.epsilon, x)

        _pattern = re.compile(r"^\[(.*), *(.*)\)((%,%missing)*)")

        def _extract_breaks(x):
            gp23 = _pattern.match(x)
            breaks_string = x if gp23 is None else gp23.group(2)
            return breaks_string

        # yapf: disable
        binning = binning.assign(
            count=lambda x: x['good'] + x['bad'],
            bad_dist=lambda x: _rm0(x['bad']) / _rm0(x['bad']).sum(),
            good_dist=lambda x: _rm0(x['good']) / _rm0(x['good']).sum()
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
        binning.rename(columns={'bin_chr': 'bin'}, inplace=True)

        return binning[[
            'variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob',
            'woe', 'bin_iv', 'total_iv', 'breaks', 'is_special_values'
        ]]


def woebin_factory(*args, **kwargs) -> WOEBin:
    return WOEBin()
