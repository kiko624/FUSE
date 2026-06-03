# FUSE: 适用于时间序列预测的自适应频时统一语义攻击方法

本项目基于 [BasicTS](https://github.com/GestaltCogTeam/BasicTS) 开发，是一个适用于时间序列预测的自适应频时统一语义攻击方法

## 🚀 快速上手

关于项目的详细环境配置、数据集准备以及基础预测模型的训练方法，请参考：
👉 **[快速上手 (Getting Started)](./tutorial/getting_started_cn.md)**

## 🛠 对抗攻击实验

项目集成了多种时间序列对抗攻击算法，包括FGSM,BIM.MI-FGSM,NI-FGSM,Fre-C&W,TCA,MAPGD,AAIM,FUSE,可以通过运行以下脚本直接开展实验：

### 1. 白盒攻击 (White-box Attack)
如果你需要对拥有完整参数信息的模型进行白盒攻击实验，请运行：
```bash
python run_whitebox_suite.py
```

### 2. 黑盒攻击 (Black-box Attack)
如果你需要对仅有输入输出交互权限的模型进行黑盒攻击实验，请运行：
```bash
python run_blackbox_suite.py
```


## 🔗 致谢

本项目核心代码框架依托于 [BasicTS](https://github.com/GestaltCogTeam/BasicTS)。感谢相关开发团队的开源贡献。
