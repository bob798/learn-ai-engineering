---
name: fix-dep
description: 诊断并修复 Python 依赖报错，包括 ModuleNotFoundError、ImportError、版本冲突、venv 路径问题。粘贴报错信息即可。
disable-model-invocation: true
allowed-tools: Bash, Read, Edit
---

# Python 依赖报错修复

报错内容：$ARGUMENTS

---

## Step 1：识别报错类型

从 $ARGUMENTS 判断报错类别：

| 报错关键词 | 类型 | 常见原因 |
|---|---|---|
| `ModuleNotFoundError` | 包未安装 | 装在了错误的 Python / venv |
| `ImportError` | 包存在但导入失败 | 版本不兼容 / 依赖缺失 |
| `Cannot install X==Y.Z` | 版本冲突 | 其他包锁定了版本 |
| `Disabling PyTorch` | 版本过低 | PyTorch < 2.4 |
| `No module named pip` | venv 损坏 | venv 未正确创建 |

## Step 2：确认当前 Python 环境

```bash
# 找项目使用的 Python
which python3
python3 --version
ls rag/code/.venv/bin/python* 2>/dev/null || echo "no venv"

# 确认包安装位置
python3 -c "import sys; print('\n'.join(sys.path))"
```

## Step 3：按类型修复

### ModuleNotFoundError
```bash
# 先确认用哪个 python 运行的脚本
# 如果用 python3 运行，就用 python3 -m pip 安装
python3 -m pip install <package>

# 如果项目有 .venv
rag/code/.venv/bin/python -m pip install <package>
# 或用 uv（更快）
uv pip install <package>
```

### 版本冲突（Cannot install）
```bash
# 查看冲突依赖树
python3 -m pip show <package>
# 忽略版本限制安装（谨慎）
python3 -m pip install <package> --no-deps
# 或升级整个依赖
python3 -m pip install <package> --upgrade
```

### PyTorch 版本过低
```bash
# 查看当前版本
python3 -c "import torch; print(torch.__version__)"
# CPU 版本升级（不需要 GPU）
python3 -m pip install torch --upgrade --index-url https://download.pytorch.org/whl/cpu
```

### venv 找不到 pip
```bash
# 用 ensurepip 重建
python3 -m ensurepip --upgrade
# 或重新创建 venv
python3 -m venv rag/code/.venv --clear
```

## Step 4：验证修复

重新运行原来报错的命令，确认错误消失。

若报错变化（出现新错误），重复 Step 1。

## Step 5：记录到项目

若此依赖需要长期维护，更新 `requirements.txt` 或脚本顶部注释：
```bash
python3 -m pip freeze | grep <package> >> requirements.txt
```
