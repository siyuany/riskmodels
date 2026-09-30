# -*- encoding: utf-8 -*-
"""
分箱核心基类模块

提供分箱器的基础类和 Mixin
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import syriskmodels.logging as logging


class WOEBin(ABC):
    """WOEBin: 分箱操作基类
    
    所有分箱类都继承自本类。分箱类型包括：
    - 细分箱 (*InitBin): 等频或等宽初始化分箱
    - 粗分箱 (*OptimBin): 在细分箱基础上简化分箱结果
    
    WOEBin 实例为可调用对象，接受 dtm、breaks、special_values 参数。
    
    分箱名约定:
    - 数值型变量：[a,b) 左闭右开区间
    - 类别变量：c1%,%c2%,%...%,%cn (用%,%拼接)
    
    切分点约定:
    - 数值型变量：区间右边界 b
    - 类别变量：与分箱名相同
    
    参数:
        eps: WOE/IV计算中替换 0 的值，默认 0.5
        **kwargs: 其他参数
    """
    
    def __init__(self, eps: float = 0.5, **kwargs):
        self.epsilon = eps
        self.kwargs = kwargs
    
    @staticmethod
    def add_missing_spl_val(dtm: pd.DataFrame, spl_val: Optional[List]) -> Optional[List]:
        """如果数据集中存在空值，则将 'missing' 加入到 special_values 中"""
        special_values = spl_val
        if dtm['value'].isnull().any():
            if spl_val is None:
                special_values = ['missing']
            elif 'missing' not in spl_val:
                special_values = ['missing'] + spl_val
        return special_values
    
    @staticmethod
    def split_vec_to_df(vec: List[Any]) -> pd.DataFrame:
        """特殊值/断点列表转 DataFrame
        
        参数:
            vec: 特殊值列表，如 ['missing', '1', '2', '3%,%4']
        
        返回:
            DataFrame，包含 bin_chr, rowid, value 三列
        
        示例:
            >>> WOEBin.split_vec_to_df(['missing', '1', '2', '3%,%4'])
               bin_chr  rowid value
            0  missing      0   NaN
            1        1      1     1
            2        2      2     2
            3    3%,%4      3     3
            4    3%,%4      3     4
        """
        assert vec is not None, 'vec cannot be None'
        vec = [str(i) for i in vec]
        a = pd.DataFrame({'bin_chr': vec}).assign(rowid=lambda x: x.index)
        b = pd.DataFrame([i.split('%,%') for i in vec], index=vec).stack().replace(
            'missing', np.nan).reset_index(name='value').rename(
                columns={'level_0': 'bin_chr'})[['bin_chr', 'value']]
        df = pd.merge(a, b, on='bin_chr')
        return df
    
    @classmethod
    def split_special_values(
        cls,
        dtm: pd.DataFrame,
        spl_val: Optional[List]
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """拆分特殊值和非特殊值数据

        W2 重构（B-4 / B-5）：不再通过 ``fillna("missing") + merge`` 匹配特殊值
        —— pandas 3 下 object↔float64 的 merge 键直接抛 ``ValueError``（B-4），
        ``astype(int)`` 也不再允许 NaN（B-5）。改为**布尔掩码 + 值→bin_chr
        映射**，语义与 pandas 2 时代保持一致：

        * ``'missing'`` 条目匹配 NaN 行，bin_chr 为原条目字符串；
        * 数值列：特殊值字符串解析为数值后与原始值做数值相等匹配；
          bin_chr 标签沿用 legacy 规则 —— 原列为整数 dtype 且特殊值不含
          NaN 时为 ``str(int(v))``（与旧 ``astype(原dtype)`` 的截断一致），
          否则为 ``str(float(v))``（如 ``-999.0``）；含 NaN 的整数列保持
          float 语义（B-5，旧实现在此路径直接抛错，无 legacy 行为可保留）；
          组合条目（``'3%,%4'``）按 legacy 语义拆分为独立分箱
          （bin_chr 为单个数值的字符串形式）；
        * 类别列：特殊值字符串与原始值直接相等匹配，组合条目整体作为一个
          bin_chr；
        * 非特殊值部分 ``dtm_ns`` 的 ``value`` 保持原 dtype（旧实现经
          fillna+merge 后再 ``astype`` 还原，两者结果一致）。

        参数:
            dtm: 输入数据 (variable, y, value 三列)
            spl_val: 特殊值列表

        返回:
            字典 {'dtm_sv': 特殊值数据（含 bin_chr/rowid 列，索引为原始 idx），
                  'dtm_ns': 非特殊值数据（索引为原始 idx）}
        """
        dtm = dtm.copy()
        dtm['idx'] = dtm.index
        spl_val = cls.add_missing_spl_val(dtm, spl_val)

        if spl_val is None:
            dtm_ns = dtm.set_index(dtm['idx'], drop=True)
            return {'dtm_sv': None, 'dtm_ns': dtm_ns}

        sv_df = cls.split_vec_to_df(spl_val)
        value_col = dtm['value']
        is_num = is_numeric_dtype(value_col)

        n = len(dtm)
        nan_mask = value_col.isna().to_numpy()
        matched = np.zeros(n, dtype=bool)
        match_bin = np.empty(n, dtype=object)
        match_rowid = np.empty(n, dtype='float64')

        sv_values = sv_df['value'].tolist()       # str；'missing' → NaN
        sv_bin_chrs = sv_df['bin_chr'].tolist()   # 原始条目字符串
        sv_rowids = sv_df['rowid'].tolist()

        if is_num:
            original_dtype = value_col.dtype
            # 与旧 astype 行为一致：非数值字符串抛 ValueError
            sv_num = [np.nan if pd.isna(v) else float(v) for v in sv_values]
            has_nan_sv = any(np.isnan(v) for v in sv_num)
            # B-5：仅当特殊值不含 NaN 且原列为整数 dtype 时才回落整数语义
            as_int = (not has_nan_sv) and pd.api.types.is_integer_dtype(
                original_dtype)
            # 统一转为 float64 做数值相等匹配（整数在 2^53 内精确表示；
            # 同时兼容 nullable Int64 等扩展 dtype）
            arr = value_col.to_numpy(dtype='float64', na_value=np.nan)

            for v, bin_name, rowid in zip(sv_num, sv_bin_chrs, sv_rowids):
                if np.isnan(v):
                    m = nan_mask                    # 'missing' 条目
                    label = bin_name
                else:
                    if as_int:
                        # 截断语义与旧 astype(int) 一致
                        v = float(int(v))
                        label = str(int(v))
                    else:
                        label = str(float(v))
                    m = (arr == v) & ~nan_mask
                match_bin[m] = label
                match_rowid[m] = float(rowid)
                matched |= m
        else:
            arr_obj = value_col.to_numpy(dtype=object)
            for v, bin_name, rowid in zip(sv_values, sv_bin_chrs, sv_rowids):
                if pd.isna(v):
                    m = nan_mask                    # 'missing' 条目
                else:
                    m = (arr_obj == v) & ~nan_mask
                match_bin[m] = bin_name
                match_rowid[m] = float(rowid)
                matched |= m

        if matched.any():
            dtm_sv = dtm.loc[matched].copy()
            dtm_sv['bin_chr'] = match_bin[matched]
            dtm_sv['rowid'] = match_rowid[matched]
            # 与 legacy 一致：特殊值部分的 NaN value 记为 'missing'（object 化）
            sv_nan = dtm_sv['value'].isna()
            if sv_nan.any():
                dtm_sv['value'] = dtm_sv['value'].astype(object)
                dtm_sv.loc[sv_nan, 'value'] = 'missing'
            dtm_sv = dtm_sv.set_index(dtm_sv['idx'], drop=True)
        else:
            dtm_sv = None

        if (~matched).any():
            dtm_ns = dtm.loc[~matched].copy()
            dtm_ns = dtm_ns.set_index(dtm_ns['idx'], drop=True)
        else:
            dtm_ns = None

        return {'dtm_sv': dtm_sv, 'dtm_ns': dtm_ns}
    
    @classmethod
    def binning(cls, dtm: pd.DataFrame, bin_chr: pd.Series) -> pd.DataFrame:
        """给定 dtm、分箱名序列生成 binning 统计表

        W2（Phase 2）向量化：旧实现对每个分组调用 Python lambda
        （``_n0``/``_n1`` = ``np.sum(x == 0/1)``），改为布尔→int64 列 +
        ``groupby.sum``（cython 聚合）。输出与旧实现完全一致：分组按
        (variable, bin_chr) 排序、``observed=False`` 保留空类别、NaN 组键
        被 groupby 丢弃、good/bad 为 int64。

        参数:
            dtm: 输入数据 (variable, y, value 三列)
            bin_chr: 每个样本对应的分箱名称（与 dtm 索引对齐的 Series）

        返回:
            binning DataFrame，包含 variable, bin_chr, good, bad 四列
        """
        tmp = pd.DataFrame({
            'variable': dtm['variable'],
            'bin_chr': bin_chr,
            '_good': (dtm['y'] == 0).astype('int64'),
            '_bad': (dtm['y'] == 1).astype('int64'),
        })
        binning = tmp.groupby(['variable', 'bin_chr'], observed=False)[
            ['_good', '_bad']
        ].sum().rename(
            columns={'_good': 'good', '_bad': 'bad'}
        ).reset_index()

        return binning

    @classmethod
    def dtm_binning_sv(cls, dtm: pd.DataFrame, spl_val: Optional[List]) -> Dict[str, Optional[pd.DataFrame]]:
        """将原数据集拆分为特殊值数据集、非特殊值数据集，并对特殊值部分做分箱统计。

        该实现保持与 legacy 版 `WOEBin.dtm_binning_sv` 行为一致：
        - 数值型变量：每个特殊数字单独成箱
        - 类别型变量：按照特殊值列表中的组合成箱
        - 返回：
            {'binning_sv': 特殊值分箱统计结果, 'ns_dtm': 非特殊值部分 dtm}
        """
        split_dtm = cls.split_special_values(dtm, spl_val)

        dtm_sv = split_dtm['dtm_sv']
        dtm_ns = split_dtm['dtm_ns']

        if dtm_sv is None or dtm_sv.shape[0] == 0:
            binning_sv = None
        else:
            binning_sv = cls.binning(dtm_sv, dtm_sv['bin_chr'])

        return {'binning_sv': binning_sv, 'ns_dtm': dtm_ns}
    
    @abstractmethod
    def woebin(self, dtm: pd.DataFrame, breaks: Optional[List] = None) -> List:
        """分箱主方法（抽象方法，子类必须实现）
        
        参数:
            dtm: 不含特殊值的数据 (variable, y, value 三列)
            breaks: 切分点列表（粗分箱使用）
        
        返回:
            切分点列表
        """
        pass
    
    def __call__(
        self,
        dtm: pd.DataFrame,
        breaks: Optional[List] = None,
        special_values: Optional[List] = None,
        max_cate_num: int = 50,
        replace_blank: Union[float, int] = np.nan
    ) -> Union[pd.DataFrame, str]:
        """执行分箱
        
        参数:
            dtm: 输入数据 (variable, y, value 三列)
            breaks: 用户指定切分点
            special_values: 特殊值列表
            max_cate_num: 最大类别数
            replace_blank: 空字符串替换值
        
        返回:
            分箱统计 DataFrame 或状态字符串 ('CONST', 'TOO_MANY_VALUES')
        """
        from syriskmodels.scorecard.constants import VariableStatus
        from syriskmodels.scorecard.utils.validation import check_uniques
        from syriskmodels.scorecard.utils.validation import replace_blank_string

        # 替换空字符串，保持 legacy 语义：'' → np.nan
        dtm = dtm.copy()
        dtm['value'] = replace_blank_string(dtm['value'])

        # 检查唯一值
        status = check_uniques(dtm['value'], max_cate_num)

        if status == VariableStatus.CONSTANT:
            return 'CONST'
        elif status == VariableStatus.TOO_MANY_CATEGORIES:
            return 'TOO_MANY_VALUES'

        # 拆分特殊值并对特殊值部分做分箱
        binning_split = self.dtm_binning_sv(dtm, special_values)
        bin_sv = binning_split['binning_sv']
        dtm_ns = binning_split['ns_dtm']

        # 非特殊值分箱
        if dtm_ns is None or len(dtm_ns) == 0:
            bin_ns = None
        else:
            if breaks is not None:
                # 使用用户指定的切分点
                bin_ns = self.binning_breaks(dtm_ns, breaks)
            else:
                # 调用 woebin 方法计算切分点
                breaks = self.woebin(dtm_ns)
                bin_ns = self.binning_breaks(dtm_ns, breaks)

        # 合并特殊值和非特殊值结果 — 保持与 legacy 一致的 pd.concat(keys=...) 方式
        bin_list = {'binning_sv': bin_sv, 'binning': bin_ns}
        binning = pd.concat(bin_list, keys=bin_list.keys())
        binning = binning.reset_index()
        binning = binning.assign(is_sv=lambda x: x.level_0 == 'binning_sv')

        # 格式化并计算统计量
        return self.binning_format(binning)

    @classmethod
    def apply(
        cls,
        dtm: pd.DataFrame,
        bin_res: pd.DataFrame,
        value: str = 'woe'
    ) -> pd.Series:
        """将单变量原始取值转换为 WOE / index / bin 值。

        该实现保持与 legacy 版 `WOEBin.apply` 行为一致，确保 `woebin_ply`、
        `woebin_psi` 等上层 API 的兼容性。
        """
        from syriskmodels.scorecard.utils.validation import replace_blank_string

        # 提取特殊值列表
        special_values = bin_res['breaks'][bin_res['is_special_values']].tolist()
        if 'missing' in special_values:
            special_values.remove('missing')
        if len(special_values) == 0:
            special_values = None

        # 提取普通分箱切分点
        breaks = bin_res['breaks'][~bin_res['is_special_values']]

        # 预处理原始值
        dtm = dtm.copy()
        dtm['value'] = replace_blank_string(dtm['value'])
        # W2（B-4/B-5 重构）：split_special_values 不再原地给调用方帧添加
        # 'idx' 列，这里显式自建（后续 merge 依赖该列对齐原始行）
        dtm['idx'] = dtm.index

        # 拆分特殊值 / 非特殊值
        split_dtm = cls.split_special_values(dtm, special_values)
        dtm_sv = split_dtm['dtm_sv']
        dtm_ns = split_dtm['dtm_ns']

        if dtm_sv is not None:
            dtm_sv = dtm_sv[['idx', 'bin_chr', 'value', 'y']]

        if dtm_ns is not None:
            break_df = cls.split_vec_to_df(breaks)
            # 数值型：使用切分点做区间分箱
            if is_numeric_dtype(dtm_ns['value']):
                break_list = ['-inf'] + list(
                    set(break_df.value.tolist()).difference(
                        {np.nan, '-inf', 'inf', 'Inf', '-Inf'}
                    )
                ) + ['inf']
                break_list = sorted(list(map(float, break_list)))
                labels = [
                    f'[{break_list[i]},{break_list[i + 1]})'
                    for i in range(len(break_list) - 1)
                ]
                dtm_ns['bin_chr'] = pd.cut(
                    dtm_ns['value'],
                    break_list,
                    right=False,
                    labels=labels
                ).astype(str)
            else:
                # 类别型：直接按映射表合并
                dtm_ns = pd.merge(dtm_ns, break_df, how='left', on='value')

            dtm_ns = dtm_ns[['idx', 'bin_chr', 'value', 'y']]

        # 合并特殊值与普通值
        new_dtm = pd.concat([dtm_sv, dtm_ns], ignore_index=True)
        dtm = pd.merge(dtm, new_dtm[['idx', 'bin_chr']], on='idx', how='left')

        # 将分箱结果映射为所需取值（woe/index/bin）
        bin_res = bin_res.copy()
        bin_res['index'] = bin_res.index
        bin_res['bin_chr'] = bin_res['bin']
        dtm = pd.merge(dtm, bin_res[['bin_chr', value]], on='bin_chr', how='left')
        dtm = dtm.set_index(dtm['idx'], drop=True)

        variable = dtm['variable'].iloc[0]
        feature_name = '_'.join([variable, value])
        dtm = dtm.rename(columns={value: feature_name})

        return dtm[feature_name]
    
    def binning_breaks(self, dtm: pd.DataFrame, breaks: List) -> pd.DataFrame:
        """按照给定的 breaks 进行分箱
        
        参数:
            dtm: 输入数据
            breaks: 切分点列表
        
        返回:
            分箱统计 DataFrame (variable, bin_chr, good, bad)
        """
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
        else:
            dtm = pd.merge(dtm, break_df, how='left', on='value')
            binning = self.binning(dtm, dtm['bin_chr'])
            # 保持分箱顺序与传入参数一致
            binning['bin_chr'] = binning['bin_chr'].astype(
                'category').cat.set_categories(
                    breaks, ordered=True)
            binning = binning.sort_values(by='bin_chr').reset_index(drop=True)
        
        return binning
    
    def binning_format(self, binning: pd.DataFrame) -> pd.DataFrame:
        """格式化分箱统计结果
        
        计算: count, count_distr, good, bad, badprob, lift, woe, bin_iv, total_iv, breaks
        
        参数:
            binning: 原始分箱统计 DataFrame
        
        返回:
            格式化后的 DataFrame
        """
        def sub0(x):
            """substitute 0"""
            return np.where(x == 0, self.epsilon, x)
        
        _pattern = re.compile(r"^\[(.*), *(.*)\)((%,%missing)*)")
        
        def _extract_breaks(x):
            gp23 = _pattern.match(x)
            breaks_string = x if gp23 is None else gp23.group(2)
            return breaks_string
        
        # 与 legacy 保持一致的链式计算
        binning = binning.assign(
            count=lambda x: x['good'] + x['bad'],
            bad_dist=lambda x: sub0(x['bad']) / sub0(x['bad']).sum(),
            good_dist=lambda x: sub0(x['good']) / sub0(x['good']).sum()
        ).assign(
            count_distr=lambda x: x['count'] / x['count'].sum(),
            badprob=lambda x: x['bad'] / x['count'],
            woe=lambda x: np.log(x['good_dist'] / x['bad_dist'])
        ).assign(
            lift=lambda x: x['badprob'] / (x['bad'].sum() / x['count'].sum()),
            bin_iv=lambda x: (x['good_dist'] - x['bad_dist']) * x['woe']
        ).assign(total_iv=lambda x: x['bin_iv'].sum())
        
        binning['breaks'] = binning['bin_chr'].apply(_extract_breaks)
        binning['is_special_values'] = binning['is_sv']
        binning['bin'] = binning['bin_chr'].astype('str')
        
        return binning[[
            'variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob',
            'lift', 'woe', 'bin_iv', 'total_iv', 'breaks', 'is_special_values'
        ]]


class InitBin(WOEBin):
    """细分箱基类
    
    用于初始化分箱，通常是等频或等宽分箱
    """
    
    @staticmethod
    def check_empty_bins(dtm: pd.DataFrame, breaks: List) -> List:
        """检查并移除空分箱
        
        参数:
            dtm: 输入数据
            breaks: 切分点列表
        
        返回:
            移除空分箱后的切分点
        """
        import re
        
        bins = pd.cut(dtm['value'], breaks, right=False)
        bin_sample_count = bins.value_counts()
        
        if np.any(bin_sample_count == 0):
            # 有空分箱，移除
            bin_sample_count = bin_sample_count[bin_sample_count != 0]
            bin_right = set([
                re.match(r'\[(.+),(.+)\)', str(i)).group(1)
                for i in bin_sample_count.index.astype('str')
                if re.match(r'\[(.+),(.+)\)', str(i))
            ]).difference({'-inf', 'inf'})
            breaks = sorted(list(map(float, ['-inf'] + list(bin_right) + ['inf'])))
        
        return breaks


class OptimBinMixin:
    """粗分箱 Mixin
    
    提供 initial_binning 方法，根据细分箱切分点生成分箱统计表
    """
    
    def initial_binning(self, dtm, breaks):
        """根据细分箱切分点生成分箱统计表
        
        参数:
            dtm: 输入数据
            breaks: 细分箱切分点
        
        返回:
            分箱统计 DataFrame
        """
        binning = self.binning_breaks(dtm, breaks)
        return self._initial_binning_finish(
            binning, is_numeric_dtype(dtm['value']))

    @staticmethod
    def _initial_binning_finish(binning, is_numeric):
        """initial_binning 的收尾步骤（W2 Phase 6 拆出，供缓存路径复用）。

        对给定 binning 表（列：variable/bin_chr/good/bad，行序 =
        ``binning_breaks`` 输出）补充 count/count_distr，并对类别型按
        badprob 降序重排 —— 与拆分前的 ``initial_binning`` 逐位一致；
        缓存路径以相同行序/dtype 的输入调用本方法，保证不稳定排序
        （quicksort）产生相同结果。
        """
        binning['count'] = binning['good'] + binning['bad']
        binning['count_distr'] = binning['count'] / binning['count'].sum()

        if not is_numeric:
            binning['badprob'] = binning['bad'] / binning['count']
            binning = binning.sort_values(
                by='badprob', ascending=False).reset_index(drop=True)

        return binning

    def initial_count_table(self, dtm, breaks, parent=None):
        """根据细分箱切分点生成粗分箱计数表（W2 Phase 2）。

        返回 :class:`~syriskmodels.scorecard.core.counts.BinCountTable`，
        行序与 ``initial_binning`` 输出**完全一致**（数值型按区间升序；
        类别型按 badprob 降序）。Tree / ChiMerge / Rule 内核只依赖该结构，
        不再在候选循环里操作原始 DataFrame。

        参数:
            dtm: 输入数据 (variable, y, value 三列)
            breaks: 细分箱切分点
            parent: 可选缓存 ``(父级 BinCountTable, 段边界)``（W2 Phase 6，
                由 :class:`ComposedWOEBin` 传入）。命中时通过段聚合构造
                计数表，避免对 ``dtm`` 重复 pd.cut/merge + groupby 全量
                扫描；构造结果与全新扫描逐位一致（校验失败自动回退扫描）。

        返回:
            BinCountTable
        """
        from syriskmodels.scorecard.core.counts import (
            BinCountTable,
            binning_from_segments,
        )

        is_num = is_numeric_dtype(dtm['value'])

        if parent is not None:
            parent_table, parent_bounds = parent
            if parent_bounds is not None:
                cached = binning_from_segments(
                    parent_table, parent_bounds, breaks, is_num)
                if cached is not None:
                    binning = self._initial_binning_finish(cached, is_num)
                    return BinCountTable.from_binning_df(
                        binning, is_numeric=is_num, epsilon=self.epsilon)

        binning = self.initial_binning(dtm, breaks)
        return BinCountTable.from_binning_df(
            binning,
            is_numeric=is_num,
            epsilon=self.epsilon,
        )


class ComposedWOEBin(WOEBin):
    """组合分箱器
    
    将多个 WOEBin 实例按顺序组合使用。

    W2 Phase 6：链路上各级粗分箱复用上一级的计数表 —— 每级内核返回
    ``(breaks, 输入计数表, 段边界)``（``woebin_with_table``），下一级的
    ``initial_count_table`` 由段聚合构造（免 pd.cut/merge + groupby 全量
    重扫），``__call__`` 的最终 ``binning_breaks`` 同样命中缓存。缓存以
    **对象同一性**（weakref）为键，仅在库内部调用链上生效；单独调用
    ``woebin(dtm, breaks)`` / 自定义分箱器（无 ``woebin_with_table``）
    自动走原始扫描路径，行为不变。
    
    参数:
        bins: WOEBin 实例列表
        **kwargs: 其他参数
    """
    
    def __init__(self, bins: List[WOEBin], **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
        # (weakref(dtm), weakref(breaks), (table, seg_bounds))；仅性能提示，
        # 不参与任何语义（weakref 不可 pickle → __getstate__ 中丢弃）
        self._counts_cache = None
    
    def woebin(self, dtm: pd.DataFrame, breaks: Optional[List] = None) -> List:
        """按顺序执行所有分箱器
        
        参数:
            dtm: 输入数据
            breaks: 初始切分点（可选）
        
        返回:
            最终切分点列表
        """
        import weakref

        current_breaks = breaks
        parent = None

        for i, binner in enumerate(self.bins):
            if i == 0 and current_breaks is None:
                # 第一个分箱器，无初始切分点
                current_breaks = binner.woebin(dtm)
                parent = None
            else:
                # 后续分箱器，使用前一个的切分点
                with_table = getattr(binner, 'woebin_with_table', None)
                if with_table is not None:
                    current_breaks, table, seg_bounds = with_table(
                        dtm, breaks=current_breaks, parent=parent)
                    parent = None if table is None else (table, seg_bounds)
                else:
                    current_breaks = binner.woebin(dtm, breaks=current_breaks)
                    parent = None

        # 为 __call__ 的最终 binning_breaks 缓存计数表（同一性校验）
        self._counts_cache = None
        if parent is not None:
            try:
                self._counts_cache = (
                    weakref.ref(dtm), weakref.ref(current_breaks), parent)
            except TypeError:
                # breaks 为不可弱引用对象（如 ndarray）→ 放弃缓存
                self._counts_cache = None

        return current_breaks

    def binning_breaks(self, dtm: pd.DataFrame, breaks: List) -> pd.DataFrame:
        """按照给定 breaks 分箱；命中 Phase 6 缓存时由段聚合直接构造。

        缓存命中条件（全部满足才使用，否则走父类原始扫描）：
        1. ``dtm`` 与本实例最近一次 ``woebin`` 收到的是**同一对象**
           （weakref 同一性；对象被回收后引用失效 → 自动 miss）；
        2. ``breaks`` 与该次 ``woebin`` 的返回是**同一对象**；
        3. ``binning_from_segments`` 的一致性校验通过（breaks 与段
           一一对应）。
        """
        cache = self._counts_cache
        if cache is not None:
            dtm_ref, breaks_ref, parent = cache
            if dtm_ref() is dtm and breaks_ref() is breaks:
                from syriskmodels.scorecard.core.counts import (
                    binning_from_segments,
                )
                cached = binning_from_segments(
                    parent[0], parent[1], breaks,
                    is_numeric_dtype(dtm['value']))
                if cached is not None:
                    return cached
        return super().binning_breaks(dtm, breaks)

    def __getstate__(self):
        state = self.__dict__.copy()
        # weakref 不可 pickle；缓存仅是性能提示，跨进程直接丢弃
        state['_counts_cache'] = None
        return state

    def __repr__(self):
        bin_names = [b.__class__.__name__ for b in self.bins]
        return f"ComposedWOEBin({bin_names})"
