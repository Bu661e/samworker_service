
## 总览

当前文件夹是 sam3worker 的核心模块代码。

# 开发过程

## 执行环境
使用uv构建环境

## 测试
使用pytest进行测试
推荐测试指令：
测试代码写在tests目录下，命名以test_开头
测试需要的输入在test/inputs目录下
测试输出在tests/runs目录下，每次测试运行都创建一个新的目录,命名以时间YYYY-MM-DD-HH-MM-SS为主


# 使用过程
## 使用
如果外面模块想使用 sam3worker中的函数或类，可以通过以下方式导入：

```python
from sam3worker import function_name
from sam3worker import ClassName
```

## 详细的API文档
docs/sam3worker_designed.md 里面描述了怎么使用sam3worker中的函数和类，以及它们的输入输出参数和返回值。
docs/ultralytics_SAM3_使用指南.md 里面描述了如何使用 ultralytics 的 SAM 模型，特别是 sam3.pt 模型，以及它的不同接口和适用场景。
