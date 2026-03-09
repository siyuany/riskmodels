# -*- encoding: utf-8 -*-
"""
分箱核心基类模块

提供分箱器的基础类和 Mixin
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import syriskmodels.logging as logging
from syriskmodels.utils import str_to_list


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
                special_values = spl_val + ['missing']
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
        
        参数:
            dtm: 输入数据 (variable, y, value 三列)
            spl_val: 特殊值列表
        
        返回:
            字典 {'dtm_sv': 特殊值数据，'dtm_ns': 非特殊值数据}
        """
        dtm['idx'] = dtm.index
        spl_val = cls.add_missing_spl_val(dtm, spl_val)
        
        if spl_val is not None:
            sv_df = cls.split_vec_to_df(spl_val)
            
            # 数值型变量特殊处理
            if is_numeric_dtype(dtm['value']):
                sv_df['value'] = sv_df['value'].astype(dtm['value'].dtypes)
                sv_df['bin_chr'] = np.where(
                    np.isnan(sv_df['value']), 
                    sv_df['bin_chr'],
                    sv_df['value'].astype(str)
                )
            
            # 数据拆分
            dtm_merge = pd.merge(
                dtm.fillna("missing"),
                sv_df[['value', 'rowid']].fillna("missing"),
                how='left',
                on='value'
            )
            dtm_sv = dtm_merge[~dtm_merge['rowid'].isna()][dtm.columns.tolist()]
            dtm_ns = dtm_merge[dtm_merge['rowid'].isna()][dtm.columns.tolist()]
        else:
            dtm_sv = None
            dtm_ns = dtm.copy()
        
        return {'dtm_sv': dtm_sv, 'dtm_ns': dtm_ns}
    
    @staticmethod
    def dtm_binning_sv(dtm: pd.DataFrame, special_values: List) -> pd.DataFrame:
        """特殊值分箱统计
        
        参数:
            dtm: 特殊值数据
            special_values: 特殊值列表
        
        返回:
            分箱统计 DataFrame
        """
        dtm = dtm.copy()
        dtm['bin_chr'] = dtm['value'].astype(str)
        dtm.loc[dtm['value'].isna(), 'bin_chr'] = 'missing'
        
        bin_sv = dtm.groupby('bin_chr')['y'].agg(
            good=lambda x: (x == 0).sum(),
            bad=lambda x: (x == 1).sum()
        ).reset_index()
        bin_sv['variable'] = dtm['variable'].iloc[0]
        
        # 标记是否为特殊值
        bin_sv['is_special_values'] = True
        
        return bin_sv[['variable', 'bin_chr', 'good', 'bad', 'is_special_values']]
    
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
        
        # 替换空字符串
        if replace_blank is not None:
            dtm = dtm.copy()
            dtm['value'] = dtm['value'].replace('', replace_blank)
        
        # 检查唯一值
        from syriskmodels.scorecard.utils.validation import check_uniques
        status = check_uniques(dtm['value'], max_cate_num)
        
        if status == VariableStatus.CONSTANT:
            return 'CONST'
        elif status == VariableStatus.TOO_MANY_CATEGORIES:
            return 'TOO_MANY_VALUES'
        
        # 拆分特殊值
        sv_result = self.split_special_values(dtm, special_values)
        dtm_sv = sv_result['dtm_sv']
        dtm_ns = sv_result['dtm_ns']
        
        # 特殊值分箱
        if dtm_sv is not None and len(dtm_sv) > 0:
            bin_sv = self.dtm_binning_sv(dtm_sv, special_values)
        else:
            bin_sv = None
        
        # 非特殊值分箱
        if breaks is not None:
            # 使用用户指定的切分点
            bin_ns = self.binning_breaks(dtm_ns, breaks)
        else:
            # 调用 woebin 方法计算切分点
            breaks = self.woebin(dtm_ns)
            bin_ns = self.binning_breaks(dtm_ns, breaks)
        
        # 合并特殊值和非特殊值结果
        if bin_sv is not None:
            binning = pd.concat([bin_ns, bin_sv], ignore_index=True)
        else:
            binning = bin_ns
        
        # 格式化并计算统计量
        return self.binning_format(binning)
    
    def binning_breaks(self, dtm: pd.DataFrame, breaks: List) -> pd.DataFrame:
        """使用指定切分点进行分箱
        
        参数:
            dtm: 输入数据
            breaks: 切分点列表
        
        返回:
            分箱统计 DataFrame
        """
        xvalue = dtm['value']
        
        if is_numeric_dtype(xvalue):
            # 数值型变量
            bins = pd.cut(xvalue, breaks, right=False, include_lowest=True)
        else:
            # 类别型变量
            bins = xvalue.astype(str)
        
        binning = dtm.assign(bin_chr=bins).groupby('bin_chr')['y'].agg(
            good=lambda x: (x == 0).sum(),
            bad=lambda x: (x == 1).sum()
        ).reset_index()
        binning['variable'] = dtm['variable'].iloc[0]
        binning['is_special_values'] = False
        
        return binning[['variable', 'bin_chr', 'good', 'bad', 'is_special_values']]
    
    def binning_format(self, binning: pd.DataFrame) -> pd.DataFrame:
        """格式化分箱统计结果
        
        计算: count, count_distr, good, bad, badprob, lift, woe, bin_iv, total_iv, breaks
        
        参数:
            binning: 原始分箱统计 DataFrame
        
        返回:
            格式化后的 DataFrame
        """
        binning = binning.copy()
        
        # 基础统计
        binning['count'] = binning['good'] + binning['bad']
        total_good = binning['good'].sum()
        total_bad = binning['bad'].sum()
        binning['count_distr'] = binning['count'] / binning['count'].sum()
        binning['badprob'] = binning['bad'] / binning['count']
        
        # 提升度
        overall_badprob = total_bad / (total_good + total_bad)
        binning['lift'] = binning['badprob'] / overall_badprob
        
        # WOE 和 IV 计算
        def sub0(x):
            return np.where(x == 0, self.epsilon, x)
        
        good_distr = sub0(binning['good']) / total_good
        bad_distr = sub0(binning['bad']) / total_bad
        
        binning['woe'] = np.log(good_distr / bad_distr)
        binning['bin_iv'] = (good_distr - bad_distr) * binning['woe']
        binning['total_iv'] = binning['bin_iv'].sum()
        
        # 提取切分点
        if is_numeric_dtype(binning['bin_chr'].dtype):
            binning['breaks'] = binning['bin_chr']
        else:
            # 尝试从分箱名提取右边界
            import re
            pattern = re.compile(r"^\[(.*), *(.*)\)")
            binning['breaks'] = binning['bin_chr'].apply(
                lambda x: pattern.match(x).group(2) if pattern.match(x) else x
            )
            # 转换为数值型（如果可能）
            try:
                binning['breaks'] = pd.to_numeric(binning['breaks'])
            except (ValueError, TypeError):
                pass
        
        # 重排和选择列
        col_order = [
            'variable', 'bin_chr', 'count', 'count_distr', 'good', 'bad',
            'badprob', 'lift', 'woe', 'bin_iv', 'total_iv', 'breaks',
            'is_special_values'
        ]
        
        return binning[col_order]


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
        
        bins = pd.cut(dtm['value'], breaks, right=False, include_lowest=True)
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
    
    提供分箱合并的通用方法
    """
    
    def merge_binning(self, binning: pd.DataFrame, node_ids: np.ndarray) -> pd.DataFrame:
        """根据节点 ID 合并分箱
        
        参数:
            binning: 分箱统计 DataFrame
            node_ids: 节点 ID 数组
        
        返回:
            合并后的 DataFrame
        """
        binning = binning.copy()
        binning['node_id'] = node_ids
        
        merged = binning.groupby('node_id').agg({
            'good': 'sum',
            'bad': 'sum',
            'variable': 'first'
        }).reset_index()
        
        # 合并分箱名
        bin_names = binning.groupby('node_id')['bin_chr'].apply(
            lambda x: '%,%'.join(x.astype(str))
        )
        merged['bin_chr'] = bin_names
        merged['is_special_values'] = False
        
        return merged[['variable', 'bin_chr', 'good', 'bad', 'is_special_values']]


class ComposedWOEBin(WOEBin):
    """组合分箱器
    
    将多个 WOEBin 实例按顺序组合使用
    
    参数:
        bins: WOEBin 实例列表
        **kwargs: 其他参数
    """
    
    def __init__(self, bins: List[WOEBin], **kwargs):
        super().__init__(**kwargs)
        self.bins = bins
    
    def woebin(self, dtm: pd.DataFrame, breaks: Optional[List] = None) -> List:
        """按顺序执行所有分箱器
        
        参数:
            dtm: 输入数据
            breaks: 初始切分点（可选）
        
        返回:
            最终切分点列表
        """
        current_breaks = breaks
        
        for i, binner in enumerate(self.bins):
            if i == 0 and current_breaks is None:
                # 第一个分箱器，无初始切分点
                current_breaks = binner.woebin(dtm)
            else:
                # 后续分箱器，使用前一个的切分点
                current_breaks = binner.woebin(dtm, breaks=current_breaks)
        
        return current_breaks
    
    def __repr__(self):
        bin_names = [b.__class__.__name__ for b in self.bins]
        return f"ComposedWOEBin({bin_names})"
