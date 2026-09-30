# -*- encoding: utf-8 -*-
"""datasets 模块单元测试。

``data/*.csv.gz`` 不入版本库，干净克隆后不存在。涉及真实数据内容的用例
在数据缺失时 **明确跳过并给出原因**；而“数据缺失应抛 ``FileNotFoundError``”
这一契约由 ``TestMissingDataContract`` 覆盖，不依赖数据文件是否存在
（通过 ``SYRISKMODELS_DATA_DIR`` 指向空目录来构造）。
"""
import pytest
import pandas as pd

from syriskmodels.datasets import load_germancredit, load_creditcard, get_data_dir

from test.conftest import CREDITCARD_FILE, GERMANCREDIT_FILE, has_data, require_data


class TestGetDataDir:
    """get_data_dir() 测试"""

    def test_returns_path(self):
        data_dir = get_data_dir()
        assert data_dir.exists(), f'数据目录不存在: {data_dir}'
        assert data_dir.is_dir()

    def test_contains_data_files(self):
        data_dir = get_data_dir()
        files = [f.name for f in data_dir.glob('*.csv.gz')]
        missing = [
            name for name in (GERMANCREDIT_FILE, CREDITCARD_FILE)
            if name not in files
        ]
        if missing:
            pytest.skip(
                f'data/ 下缺少 {missing}（数据文件不入版本库）；'
                f'实际存在: {sorted(files)}'
            )
        assert GERMANCREDIT_FILE in files
        assert CREDITCARD_FILE in files


class TestLoadGermancredit:
    """load_germancredit() 测试"""

    def test_returns_dataframe(self):
        require_data(GERMANCREDIT_FILE)
        df = load_germancredit()
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        require_data(GERMANCREDIT_FILE)
        df = load_germancredit()
        assert df.shape[0] == 1000
        assert df.shape[1] == 21

    def test_target_column_exists(self):
        require_data(GERMANCREDIT_FILE)
        df = load_germancredit()
        assert 'creditability' in df.columns

    def test_target_values(self):
        require_data(GERMANCREDIT_FILE)
        df = load_germancredit()
        assert set(df['creditability'].unique()) == {0, 1}


@pytest.mark.slow
class TestLoadCreditcard:
    """load_creditcard() 测试（数据文件大，解压读取约 1s，标记 slow）"""

    def test_returns_dataframe(self):
        require_data(CREDITCARD_FILE)
        df = load_creditcard()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        require_data(CREDITCARD_FILE)
        df = load_creditcard()
        assert 'Time' in df.columns
        assert 'Class' in df.columns
        assert 'V1' in df.columns

    def test_target_values(self):
        require_data(CREDITCARD_FILE)
        df = load_creditcard()
        assert set(df['Class'].unique()) == {0, 1}


class TestMissingDataContract:
    """数据文件缺失时的行为契约（不依赖仓库是否带数据）。"""

    def test_germancredit_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SYRISKMODELS_DATA_DIR', str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_germancredit()

    def test_creditcard_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv('SYRISKMODELS_DATA_DIR', str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_creditcard()

    def test_datasets_present_flag_matches_repo_state(self):
        """``has_data`` 探测结果应与 ``test/`` 下的软链接状态一致。"""
        from test import conftest
        for name in (GERMANCREDIT_FILE, CREDITCARD_FILE):
            assert has_data(name) == conftest.data_file_path(name).is_file()
