# GSM8K 四组复现核心（第三步离线验收）

独立实现 Du et al. 的 GSM8K 方法结构，作者仓库与原 QA 平台业务代码保持原样。本目录包含独立 Python 3.12.10 环境，SDK 为 openai 1.109.1，全部实际依赖见 requirements-repro.txt。

第二步执行核心和第三步集中离线验收已完成：原验收 42 项测试通过，10 道训练调试题的四组 CLI 检查完成 40 项任务、150 次假模型生成；重复恢复不新增调用。2026-11-10 失败恢复修复与流式传输支持后共 66 项测试通过。第四步训练题真实验证已完成：DeepSeek 主臂 10 题调试与 30 题校准、Qwen3.5 单题与 30 题校准；Qwen 准确率高于预定弱臂上限，本轮仅推进 DeepSeek 主臂。第五步平台接入和正式冻结尚未开始。假模型与训练题分数不能作为正式 test 复现效果。原验收见 `docs/离线自检-20261109.md` 和 `verification-step3.json`，本轮修复见 `docs/失败恢复修复-20261110.md`，第四步记录见 `docs/第四步训练验证-20261110.md`。

## 已有功能

- Single / Reflection / Majority / Debate 统一执行，正常生成次数为 1 / 2 / 6 / 6。
- 作者三处提示词逐字保留；串行调用，每轮只读取同伴上一轮快照，保留自身历史。
- 固定 train 调试10题、train校准30题、test正式100题及哈希；不修改原始数据。
- 作者脚本原样评分（隔离其全局变量）、fixed评分、平票判错与精确比较两个独立敏感性变体。
- fixed v2 按精确数值合并 `1`、`1.0`、`1.00` 等价票；比较容差不用于合并不同数值。作者原样评分继续保留其原投票行为，版本写入新配置与报告。
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

## 失败诊断与预算核对（2026-11-10 修复）

HTTP 失败现在保存供应商错误码、消息、类型、参数位置及可用的 Request ID；错误字段脱敏并限长，写入调用数据库及 `calls.jsonl`，终端摘要增加 `last_failure`。单字符或含空白的 Key 会在构造 SDK 前拒绝。旧运行缺失的错误消息无法补回。

`budget-status --campaign-id <id>` 只读查看账本。`budget-reconcile --campaign-id <id> --request-id <本地调用编号> --actual-cny <已核实金额> --evidence <核对依据>` 默认只预览；增加 `--confirm-billing` 才备份账本并原子写入金额及核对审计记录。仅处理 `unknown`，不处理活动请求、不重置上限、不自动推断零费用，也不启动模型。核对命令不读取 Key。

源码修复后，应保留旧失败目录，按原配置在新目录重新 `plan`，继续共用原 `campaign_id`。详细边界和操作见 `docs/失败恢复修复-20261110.md`。

## 验证与后续

推荐使用完整离线验收入口，自动校验原始数据/作者文件，运行全部测试与 CLI 保存、恢复、报告、重新评分流程。每个 Python 子进程均启用 socket/DNS 网络拦截；SDK 路径仅使用 MockTransport：

~~~powershell
$env:GSM_REPRO_DISABLE_MODEL_CALLS = '1'
.\.venv\Scripts\python.exe -B scripts\verify_offline.py
~~~

产物默认写入新的 `runs/step3-offline-<时间>/`，包含验证 JSON、测试日志和完整假模型轨迹。原作者目录和数据目录默认与本仓库并列，也可通过 `--author`、`--data` 显式指定。单独运行单元测试：

~~~powershell
$env:GSM_REPRO_DISABLE_MODEL_CALLS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

测试统一拦截socket/DNS；SDK验证仅使用httpx.MockTransport，不连接本地或外部模型服务。当前测试针对新核心，不运行原QA完整回归。独立环境可用 requirements-repro.txt重建；作者/数据的核对记录在author_reference/provenance.json。

下一步是第四步训练题真实验证，需要用户明确启动真实调用，并核验价格、供应商参数、seed 和关闭 reasoning 的实际效果。离线通过不会把 `parameters_verified` 或价格确认改为 true。平台接入留在第五步，正式冻结和平台100题运行依次在第六、七步。
