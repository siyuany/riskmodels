# AGENTS.md — riskmodels 开发约定

本文件面向在此仓库工作的 AI agent，记录**必须遵守**的开发约定与本项目背景。
（本文档已提交到版本库，为团队共享约定。）

## 0. 首要规则

**开发前必须先写开发计划，并等待人工审阅通过，之后才能动手写代码。**
未经审阅通过的计划，不得开始实现。详见第 3 节。

## 1. 项目概览

`riskmodels`（PyPI 包名 `syriskmodels`）是信用风险模型工具包，覆盖：

- 数据探索（`syriskmodels.detector`）
- 变量分箱（`syriskmodels.scorecard`）
- 逻辑回归建模（`syriskmodels.models`、`syriskmodels.contrib`）
- 评分卡转换与规则挖掘（`syriskmodels.rule`）
- 模型评估（`syriskmodels.evaluate`：KS、gains table）

当前版本见 `pyproject.toml` 的 `version` 字段与 `src/syriskmodels/__init__.py` 的 `__version__`，两处需保持一致。

## 2. Python 环境约定

**统一使用 `~/.venvs/jep` 虚拟环境，所有依赖安装到该环境，不要新建其他 venv。**

```bash
# 绝对路径解释器（推荐，避免依赖 shell 激活状态）
~/.venvs/jep/bin/python -c "import syriskmodels; print(syriskmodels.__version__)"
~/.venvs/jep/bin/python -m pip install <package>
~/.venvs/jep/bin/python -m pytest test/ -q

# 或者显式激活
source ~/.venvs/jep/bin/activate
```

规则：

- 禁止使用系统 Python 或裸 `pip` / `python`，一律走 `~/.venvs/jep/bin/` 下的可执行文件。
- **必须用 `python -m pip` / `python -m pytest` 形式**，避免 PATH 指向其他环境的同名命令。
- 需要新增依赖时：安装到 `~/.venvs/jep`，并同步登记到 `pyproject.toml` 的 `dependencies`（注明合理的下限约束）。

## 3. Git Flow 与开发工作流

仓库使用 git flow 管理模式：

| 角色 | 分支 |
| --- | --- |
| 生产 | `main` |
| 集成分支 | `develop`（日常开发基线） |
| 功能 | `feature/<name>` |
| 缺陷修复 | `bugfix/<name>` |
| 发布 | `release/<version>` |
| 紧急修复 | `hotfix/<name>` |

> 注：本机的 `git flow` CLI 未安装，但相关配置项（`gitflow.branch.*`、`gitflow.prefix.*`）已就位。
> 请直接使用等价的 `git checkout -b` / `merge --no-ff` 命令，分支命名严格遵循上表前缀。

### 标准流程

1. **确认基线**：`git checkout develop && git pull`
2. **开分支**：`git checkout -b feature/<name>`
3. **写计划**：在**新分支下**编写 `docs/plans/<name>.md`（见第 4 节），提交该计划
4. **等待审阅**：把计划交给人工确认；**审阅通过前不写实现代码**
5. **实施**：按计划编码，同步补测试
6. **验证**：`~/.venvs/jep/bin/python -m pytest test/ -q` 全部通过
7. **提交**：遵循 Conventional Commits（见下）
8. **合并**：`git checkout develop && git merge --no-ff feature/<name>`，测试通过后删除该 feature 分支

禁止直接向 `main` 或 `develop` 提交功能代码；`main` 只接收来自 `release/*` / `hotfix/*` 的合并。

### 提交信息

采用 Conventional Commits：`<type>(<scope>): <description>`

- 常用 `type`：`feat` / `fix` / `refactor` / `test` / `docs` / `chore` / `perf`
- 历史提交为中英文混用，两者皆可，同一提交内保持一致
- 示例：`fix(scorecard): correct woebin monotonic constraint`、`refactor: replace .values with .to_numpy()`

### 冲突处理

合并前先 `git rebase develop` 到最新基线，解决冲突后再提交，避免把冲突带入合并节点。

## 4. 开发计划存放约定

- 目录：`docs/plans/`
- 文件名：与被开发内容同名，如 `docs/plans/scorecard-woebin-monotonic.md`
- 该目录**不纳入版本控制**（`.gitignore` 已忽略 `docs/`，`plans/` 亦显式忽略）。

计划至少包含：

1. **背景 / 问题**：为什么要做，当前痛点是什么
2. **目标与非目标**：明确边界，写清本次不做什么
3. **方案设计**：改动哪些模块、关键接口签名、算法或数据结构选择
4. **兼容性影响**：是否破坏现有公开 API、是否影响已有测试与用户代码
5. **实施步骤**：可逐条勾选的清单
6. **测试计划**：新增/修改哪些测试用例，如何验证
7. **风险与回滚**：已知风险点，以及出问题时的回退方式

## 5. 项目结构

```
src/syriskmodels/           # 包源码
├── __init__.py             # 公开 API 导出，版本号
├── scorecard/              # 分箱与评分卡核心（api / bins / core / utils）
├── detector.py             # 变量分布探索
├── models.py               # stepwise_lr 等建模工具
├── evaluate.py             # KS、gains_table、model_eval
├── datasets.py             # 内置数据集加载（load_creditcard / load_germancredit）
├── rule.py                 # 规则挖掘（RIPPER 等）
├── contrib/                # build_scorecard、var_select
├── logging.py / utils.py   # 日志与通用工具
└── scorecard_legacy.py     # 旧版实现，改动前先确认是否仍被引用

test/                       # pytest 测试（test_*.py 平铺 + test_scorecard/ 子目录）
data/                       # 数据集（.csv.gz）
docs/plans/                 # 开发计划（本地，不提交）
docs/references/            # 参考论文等本地资料（不提交）
meta/                       # 本地笔记（不提交）
```

## 6. 测试约定

- 测试框架 pytest，测试文件放在 `test/`，命名 `test_*.py`
- 测试数据位于 `data/`，`test/` 下通过相对软链接访问（`creditcard.csv.gz`、`germancredit.csv.gz`）
- **不要给测试添加 `sys.path` hack**（历史上已清理过），依赖包以可编辑方式安装
- 新增功能必须带测试；修复缺陷必须先有能复现该缺陷的测试

```bash
~/.venvs/jep/bin/python -m pytest test/ -q              # 全量
~/.venvs/jep/bin/python -m pytest test/test_detect.py -q # 单文件
```

## 7. 依赖与兼容性要点

`pyproject.toml` 要求 Python >= 3.11，依赖 pandas >= 3.0、numpy >= 2.0 等。历史上已为升级做过以下适配，**新代码必须沿用**：

| 禁止写法 | 应改为 | 原因 |
| --- | --- | --- |
| `x is np.nan` | `pd.isna(x)` | 身份比较不可靠 |
| `.values` | `.to_numpy()` | pandas 3.0 兼容 |
| `df.method(..., inplace=True)` | `df = df.method(...)` | pandas 3.0 Copy-on-Write |
| `pandas.core.*` 内部导入 | 公开 API | 内部结构不稳定 |
| 旧版 sklearn 参数名 | 现行参数名（如 `is_classifier`） | sklearn 1.8 兼容 |

## 8. 输出物与生成文件

- 构建产物、统计报告等生成物写入 `dist/`（已在 `.gitignore` 中忽略），不要散落在仓库根目录
- 临时脚本、一次性数据提取脚本不要提交到仓库根目录，放入 `docs/` 或 `meta/`（均为本地忽略目录），或使用后删除
- 提交前检查 `git status`，确认没有误加 `*.xlsx`、`*.bak`、数据集软链接等被忽略的文件
