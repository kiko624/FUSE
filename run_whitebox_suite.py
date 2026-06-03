import os
import json
import torch
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from easydict import EasyDict

# 实验任务矩阵
# method: 算法函数名
# mode: targeted / untargeted
# target: trend_reversal /trend_enhance/ freq_removal /freq_enhance/seasonal/ original
# ==================== 1. 配置与路径 ====================
# 请确保这里的导入路径指向你当前的配置文件
from baselines.DLinear.ETTh1_adv import CFG

# 权重路径
CHECKPOINT_PATH = "checkp/DLinear/ETTh1_100_96_96/ff7af221a8c2e062f6a90f21fc6d9a4e/DLinear_best_val_MAE.pt"

# 实验任务矩阵
WHITEBOX_TASKS = [

    {"method": "fgsm_attack", "mode": "untargeted", "target": "original"},
    {"method": "bim_attack", "mode": "untargeted", "target": "original"},
    {"method": "basic_cw_attack", "mode": "untargeted", "target": "original"},
    {"method": "fre_cw_attack", "mode": "untargeted", "target": "original"},
    {"method": "mi_fgsm_attack", "mode": "untargeted", "target": "original"},
    {"method": "ni_fgsm_attack", "mode": "untargeted", "target": "original"},
    {"method": "tca_attack", "mode": "untargeted", "target": "original"},
    {"method": "mapgd_tsf_attack", "mode": "untargeted", "target": "original"},
    {"method": "aaim_attack", "mode": "untargeted", "target": "original"},
    {"method": "fuse_attack", "mode": "untargeted", "target": "original"},
    # 你可以继续添加其他任务
]


def run_whitebox_automation():
    base_save_dir = CFG.TRAIN.CKPT_SAVE_DIR
    summary_file = "untargeted_DLinear_ETTh1_whitebox_attack_summary.csv"

    # 🌟 [关键修改]: 尝试加载已有的 CSV 文件
    if os.path.exists(summary_file):
        try:
            all_results = pd.read_csv(summary_file).to_dict('records')
            print(f"📊 发现已有汇总表，加载了 {len(all_results)} 条实验记录。")
        except Exception as e:
            print(f"⚠️ 读取汇总表失败，将从头开始: {e}")
            all_results = []
    else:
        print("📝 未发现汇总表，将创建新表。")
        all_results = []

    for task in WHITEBOX_TASKS:
        algo = task["method"]
        mode = task["mode"]
        target_type = task["target"]

        # 🌟 [关键修改]: 检查当前任务是否已经完成过
        is_finished = any(
            r['Algorithm'] == algo and
            r['Mode'] == mode and
            r['Target'] == target_type
            for r in all_results
        )

        if is_finished:
            print(f"⏩ 跳过已完成任务: {algo} | {mode} | {target_type}")
            continue

        # 构造子文件夹名称
        short_algo = algo.replace('bim_attack_time_domain_', '').replace('_attack', '')
        exp_name = f"white_{short_algo}_{mode}_{target_type}"

        print(f"\n" + "=" * 70)
        print(f"🚀 正在启动白盒攻击: {algo}")
        print("=" * 70)

        # 1. 深度克隆配置
        curr_cfg = EasyDict(deepcopy(CFG))
        curr_cfg.ATTACK.ENABLED = True
        curr_cfg.ATTACK.METHOD = algo
        curr_cfg.ATTACK.MODE = mode
        curr_cfg.ATTACK.TARGET_TYPE = target_type

        # 修复 EasyTorch 要求的键
        curr_cfg['MODEL.NAME'] = curr_cfg.MODEL.NAME
        curr_cfg['MD5'] = exp_name
        curr_cfg.TRAIN.CKPT_SAVE_DIR = base_save_dir

        actual_dir = os.path.join(base_save_dir, exp_name)
        os.makedirs(actual_dir, exist_ok=True)

        try:
            # 2. 初始化 Runner
            runner = curr_cfg.RUNNER(curr_cfg)
            runner.init_test(curr_cfg)

            if os.path.exists(CHECKPOINT_PATH):
                runner.load_model(ckpt_path=CHECKPOINT_PATH)
            else:
                print(f"❌ 错误: 找不到权重文件 {CHECKPOINT_PATH}")
                continue

            # 3. 执行攻击测试
            test_return = runner.test(save_results=True, save_metrics=True)

            # 4. 收集指标
            metrics = test_return.get("metrics_summary", {})
            entry = {
                "Algorithm": algo,
                "Mode": mode,
                "Target": target_type,
                "MAE": metrics.get("MAE", 0),
                "MSE": metrics.get("MSE", 0),
                "WAPE": metrics.get("WAPE", 0),
                "Success_Count": test_return.get("count", 0),
                "Success_Rate": metrics.get("success_rate", 0.0),
                "L1": metrics.get("L1", 0),  # <--- 对应 Runner 中修改后的 L1
                "L2": metrics.get("L2", 0),
                "Time_Cost": f"{metrics.get('attack_time', 0):.2f}s",  # <--- 记录运行时间
                "Saved_In": exp_name
            }
            all_results.append(entry)

            # 实时更新汇总 CSV
            pd.DataFrame(all_results).to_csv(summary_file, index=False)
            print(f"🏁 任务完成: MAE={entry['MAE']:.4f}, Time={entry['Time_Cost']}")


        except Exception as e:
            print(f"❌ 实验失败: {exp_name}")
            import traceback
            traceback.print_exc()

        finally:
            if 'runner' in locals():
                del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n✨ 所有白盒实验已执行完毕！汇总表: {summary_file}")


if __name__ == "__main__":
    run_whitebox_automation()