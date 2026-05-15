# Stock Analysis - A股暴涨潜力预测系统

基于历史暴涨前技术特征匹配，筛选当前市场潜在暴涨标的。

## 功能

- **历史分析**: 回测过去2年A股市场中月涨幅超100%的股票，提取暴涨前的技术特征画像
- **潜力预测**: 对当前全市场股票进行相似度评分，筛选技术形态与历史暴涨前相似的标的
- **Web Dashboard**: Flask + Plotly 可视化展示预测结果、特征对比和个股K线

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 采集数据
python run.py collect

# 分析历史暴涨
python run.py analyze

# 运行预测
python run.py predict

# 启动Web面板
python run.py web

# 一键执行全部步骤
python run.py all
```

启动后访问 http://127.0.0.1:5000

## 配置

编辑 `config.py`：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `USE_PROXY` | 是否使用代理（7890端口） | `False` |
| `LOOKBACK_YEARS` | 历史回测年限 | `2` |
| `SURGE_THRESHOLD` | 暴涨阈值（涨幅倍数） | `1.0` (100%) |
| `SURGE_WINDOW_DAYS` | 暴涨窗口（交易日） | `20` |

## 预测方法

提取当前A股近60个交易日的技术指标（RSI、波动率、量比、布林带位置等），与历史上暴涨股票在暴涨前的技术特征进行加权高斯相似度匹配，输出 0-100 的匹配评分。

⚠️ 此系统仅供研究参考，不构成投资建议。历史规律不代表未来表现。

## 技术栈

Python / Flask / Pandas / Plotly / scikit-learn
