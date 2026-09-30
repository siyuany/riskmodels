# -*- encoding: utf-8 -*-
"""
WOEBin 工厂模块

提供分箱器的注册和创建功能
"""
import inspect
from typing import List, Union, Type

import syriskmodels.logging as logging
from syriskmodels.utils import str_to_list
from syriskmodels.scorecard.core.base import WOEBin, ComposedWOEBin


class WOEBinFactory:
    """WOEBin 工厂类
    
    提供分箱器的注册和创建功能。支持将多个分箱器组合使用。
    
    使用示例:
        >>> woebin = WOEBinFactory.build(['quantile', 'tree'])
        >>> woebin(dtm)
        
        或使用类名:
        >>> woebin = WOEBinFactory.build([QuantileInitBin, TreeOptimBin])
        >>> woebin(dtm)
    
    注册新分箱器:
        >>> @WOEBinFactory.register('custom')
        ... class CustomBin(WOEBin):
        ...     def woebin(self, dtm, breaks=None):
        ...         # 实现分箱逻辑
        ...         pass
    """
    
    __woebin_class_mapping = {}

    @staticmethod
    def _filter_kwargs_for(bin_class: Type[WOEBin], kwargs: dict) -> dict:
        """按构造函数签名过滤 kwargs（B-6）。

        * 构造函数接受 ``**kwargs``（VAR_KEYWORD）→ 全量透传（legacy 行为，
          多余参数由基类 ``WOEBin.kwargs`` 收纳，不会抛 TypeError）；
        * 否则只保留构造函数**显式声明**的参数 —— 全局共享的 kwargs
          （如 ``initial_bins`` 只属于细分箱器）不再无差别透传给所有
          binner，避免组合 ``rule`` 等严格签名的分箱器时报
          ``TypeError: unexpected keyword argument``。
        """
        try:
            params = inspect.signature(bin_class.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - 内建类等特殊对象
            return dict(kwargs)

        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values()):
            return dict(kwargs)

        filtered = {k: v for k, v in kwargs.items() if k in params}
        dropped = sorted(set(kwargs) - set(filtered))
        if dropped:
            logging.debug(
                f'{bin_class.__name__} 的构造函数不接受参数 {dropped}，已忽略')
        return filtered
    
    @classmethod
    def register(cls, names: Union[str, List[str]]):
        """注册分箱类的装饰器
        
        对分箱类使用该装饰器并指定注册名称后，在 `build` 方法中就可以使用
        注册名称替代类名。
        
        参数:
            names: str 或 list[str]，分箱类的注册名称
        
        返回:
            装饰器函数
        
        示例:
            >>> @WOEBinFactory.register(['chi2', 'chimerge'])
            ... class ChiMergeOptimBin(WOEBin):
            ...     pass
        """
        names = str_to_list(names)
        
        def wrapped(bin_class):
            if not issubclass(bin_class, WOEBin):
                raise TypeError(f'类 {bin_class} 不是 WOEBin 子类，无法注册')
            
            for name in names:
                if name in cls.__woebin_class_mapping.keys():
                    raise KeyError(f'名称 {name} 已存在，'
                                   f'类 {bin_class.__name__} 不能注册为 {name}')
                else:
                    cls.__woebin_class_mapping[name] = bin_class
            
            return bin_class
        
        return wrapped
    
    @classmethod
    def get_binner(cls, bin_class: Union[str, Type[WOEBin], WOEBin], **kwargs) -> WOEBin:
        """获取分箱器实例
        
        参数:
            bin_class: 分箱类名（字符串）、类或实例
            **kwargs: 初始化参数
        
        返回:
            WOEBin 实例
        
        异常:
            KeyError: 字符串类名未注册
            TypeError: 不是 WOEBin 实例或子类
        """
        if isinstance(bin_class, str):
            try:
                bin_class = cls.__woebin_class_mapping[bin_class]
            except KeyError:
                raise KeyError(f'分箱方法 {bin_class} 未注册！')
        
        if isinstance(bin_class, WOEBin):
            binner = bin_class
        elif isinstance(bin_class, type) and issubclass(bin_class, WOEBin):
            # B-6：按构造签名过滤 kwargs，实例原样返回（kwargs 不覆盖实例配置）
            binner = bin_class(**cls._filter_kwargs_for(bin_class, kwargs))
        else:
            raise TypeError(f'类 {bin_class} 不是 WOEBin 实例或子类')
        
        return binner
    
    @classmethod
    def build(cls, bin_classes: List[Union[str, Type[WOEBin], WOEBin]], **kwargs) -> ComposedWOEBin:
        """将多个分箱器组装为一个 ComposedWOEBin
        
        参数:
            bin_classes: WOEBin 子类、实例或注册名列表
            **kwargs: 传递给分箱器初始化的关键字参数
        
        返回:
            ComposedWOEBin 实例
        
        示例:
            >>> woe_bin = WOEBinFactory.build(
            ...     ['quantile', 'tree'],
            ...     initial_bins=20,
            ...     bin_num_limit=8,
            ...     min_iv_inc=0.1,
            ...     count_distr_limit=0.05
            ... )
            >>> woe_bin
            ComposedWOEBin(['QuantileInitBin', 'TreeOptimBin'])
        """
        bin_objects = [cls.get_binner(bin_cls, **kwargs) for bin_cls in bin_classes]
        return ComposedWOEBin(bin_objects, **kwargs)
    
    @classmethod
    def get_registered_names(cls) -> List[str]:
        """获取所有已注册的分箱器名称
        
        返回:
            注册名称列表
        """
        return list(cls.__woebin_class_mapping.keys())
    
    @classmethod
    def get_class(cls, name: str) -> Type[WOEBin]:
        """根据注册名称获取分箱类
        
        参数:
            name: 注册名称
        
        返回:
            WOEBin 子类
        
        异常:
            KeyError: 名称未注册
        """
        try:
            return cls.__woebin_class_mapping[name]
        except KeyError:
            raise KeyError(f'分箱方法 {name} 未注册')
