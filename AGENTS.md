比赛介绍：README_CN.md
第二赛季提交说明：UPLOAD_README.md
优化要求：
1. 不允许使用量化、稀疏化等等，任何修改模型或者参数的优化方式。
2. 评测方式为单batch评测，指标包括TTFT、吞吐、准确度。具体参考 README_CN.md
3. GPU为单张 NVIDIA A800 80GB PCIe
参考项目（部分）：
./reference/vllm/
./reference/sglang/
./reference/llama.cpp/
开发环境：所有开发完全在./venv环境中进行。
优化习惯：将每一步的优化动作和测试结果记录在result.csv中。
官方信息更新：
1. 评测环境机器没有配置连接外网。安装受限。（但是本地已经安装了flash-attn2.8.3了）。
工作建议：
1. 在一项优化完成后，commit提交，方便后续查阅和回滚。
2. 运行bench时必须使用TQDM_DISABLE=1，不然肯定stdout阻塞导致benchmark卡住。
3. benchmark的JSON测试结果统一存放在results/目录中，便于管理和追踪。
优化文档：docs/README.md（包含优化计划、日志、踩坑记录、Trace分析、推理引擎分析等文档索引）