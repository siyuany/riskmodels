# -*- encoding: utf-8 -*-
"""
WOEBin 工厂模块

提供分箱器的注册和创建功能
"""
from typing import List, Union, Type
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
            binner = bin_class(**kwargs)
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
