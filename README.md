# GSM8K 四组复现核心（第二步）

独立实现 Du et al. 的 GSM8K 方法结构，作者仓库与原 QA 平台业务代码保持原样。本目录包含独立 Python 3.12.10 环境，SDK 为 openai 1.109.1，全部实际依赖见 requirements-repro.txt。

当前只完成第二步实现与基本离线验证。还没有正式完成第三步集中验收、第四步真实模型验证或第五步平台接入；假模型分数不能作为论文复现效果。

## 已有功能

- Single / Reflection / Majority / Debate 统一执行，正常生成次数为 1 / 2 / 6 / 6。
- 作者三处提示词逐字保留；串行调用，每轮只读取同伴上一轮快照，保留自身历史。
- 固定 train 调试10题、train校准30题、test正式100题及哈希；不修改原始数据。
- 作者脚本原样评分（隔离其全局变量）、fixed评分、平票判错与精确比较两个独立敏感性变体。
- 每次实际尝试先保存请求，再保存回复、状态、用量、耗时和关联编号；不把标准答案传给模型。
- 固定本地执行配置、源码/依赖指纹与逐调用seed清单；运行后条件改变时拒绝继续旧任务。
- 逐题持久化；resume只继续pending任务，failed/interrupted不自动重跑。结果不明请求保留unknown，全部回复已落盘但终态未保存时可核对恢复。
- SDK内置重试关闭，已知可重试错误最多补1次；超时或发送结果不明不自动重发，不依据答案是否正确重试。
- 真实费用使用跨运行、跨模型共享的campaign账本，按CNY保守预留；未知费用暂停。保留预算记录不等于保证供应商最终账单绝不超额。

## 本地使用（默认假模型）

在此目录执行下列 PowerShell 命令。示例输出目录须为一个未使用的新目录：

~~~powershell
$env:GSM_REPRO_DISABLE_MODEL_CALLS = '1'
.\.venv\Scripts\python.exe -m gsm_repro prepare
.\.venv\Scripts\python.exe -m gsm_repro plan --run-dir runs\my-offline-check --limit 2
.\.venv\Scripts\python.exe -m gsm_repro run --run-dir runs\my-offline-check --max-tasks 2
.\.venv\Scripts\python.exe -m gsm_repro resume --run-dir runs\my-offline-check
.\.venv\Scripts\python.exe -m gsm_repro report --run-dir runs\my-offline-check
~~~

plan仅保存本地配置，不读取凭据或调用模型。默认清单为train_debug_10，默认后端fake；--limit只裁剪该次开发运行，不覆盖三份原清单。输出目录含：

- profile.json：冻结的运行配置、选题、提示词/源码/环境指纹、逐调用seed。
- manifest.json：该次使用的完整清单副本，标准答案仅供评分。
- results.sqlite3：任务与每次尝试的持久化记录。
- results.jsonl / results.csv：逐题最终回答、四种评分及状态。
- calls.jsonl：逐智能体、逐轮、逐尝试的请求与回复轨迹；usage未知保留null。
- report.json：固定分母的汇总、失败、截断、请求次数、用量与耗时；未结束的报告标为partial。

status可只读查看任务。score可对已保存输出另存评分，不覆盖原始记录，也不新增模型调用：

~~~powershell
.\.venv\Scripts\python.exe -m gsm_repro status --run-dir runs\my-offline-check
.\.venv\Scripts\python.exe -m gsm_repro score --input runs\my-offline-check\results.jsonl --output runs\my-offline-check\rescored.json
~~~

## 真实调用入口的边界

development.real.example.json只是待配置示例，本次没有执行它。未来第四步明确启动后才复制为新配置，补齐已核验价格、来源/日期及参数条件；使用专用环境变量GSM_REPRO_API_KEY和GSM_REPRO_BASE_URL。程序不加载任何.env，不读取原QA项目密钥，不导出凭据，输出只保存端点摘要。

真实执行同时要求backend=openai_compatible、显式--allow-model-calls、进程未设置GSM_REPRO_DISABLE_MODEL_CALLS=1，并通过价格/预算检查。正式test还要求formal阶段、已验证参数、正式条件哈希和--allow-test-evaluation；开发配置不能拿test启动。重试、探测、校准和正式运行必须共用本轮campaign_id，不能靠新目录重置20元预算。

供应商seed、关闭reasoning的效果、版本身份及费用仍需训练题真实验证。遇到不支持的参数不静默降级；reasoning_content仅记录缺失/null/空/非空状态，不保存独立隐藏reasoning文本。未知费用须核对后再处理，不自动重发。

## 验证与后续

~~~powershell
$env:GSM_REPRO_DISABLE_MODEL_CALLS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

测试统一拦截socket/DNS；SDK验证仅使用httpx.MockTransport，不连接本地或外部模型服务。当前测试针对新核心，不运行原QA完整回归。独立环境可用 requirements-repro.txt重建；作者/数据的核对记录在author_reference/provenance.json。

下一步是第三步的集中离线验收，然后才是第四步训练题真实验证；平台接入留在第五步，正式冻结和平台100题运行依次在第六、七步。
