import os
import json
import torch
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from easydict import EasyDict

# ==================== 1. 配置与路径 ====================
# A. 导入目标模型的配置文件 (你要攻击哪个模型，就导哪个)
from baselines.PatchTST.ETTh1_adv import CFG

# B. 目标模型的权重路径
TARGET_MODEL_CKPT = "checkp/PatchTST/ETTh1_100_96_96/f4550737e34ff13e204c5c1ec4b936bf/PatchTST_best_val_MAE.pt"

# C. 干净数据的 Baseline CSV (必须先跑一次 test_pp 得到，用于对比成功率)
# 如果你没有这个文件，脚本会自动运行一次生成它
CLEAN_BASELINE_CSV = "checkp/PatchTST/ETTh1_100_96_96/baseline_run/test_losses_per_sample.csv"

# D. 待测试的黑盒任务 (对抗样本来源)
# 路径请指向你白盒脚本生成的那些 test_results.npz
BLACKBOX_SOURCES = [
    {"name": "fgsm_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_fgsm_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "bim_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_bim_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "fre_cw_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_fre_cw_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "mi_fgsm_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_mi_fgsm_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "ni_fgsm_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_ni_fgsm_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "tca_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_tca_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "mapgd_tsf_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_mapgd_tsf_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "aaim_attack",
     "path": "checkp/DLinear/ETTh1_100_96_96/white_aaim_targeted_freq_removal/test_results.npz",
     "mode": "targeted"},
    {"name": "fuse_attack",
     "path": "/root/autodl-tmp/checkp/DLinear/ETTh1_100_96_96/white_fuse_attack_targeted_freq_removal/test_results.npz",
     "mode": "targeted"}
]


# ==================== 2. 核心函数 ====================

def get_success_count(baseline_csv, current_csv, mode):
    """复用你的 compare_losses 逻辑计算成功数"""
    df_base = pd.read_csv(baseline_csv)
    df_adv = pd.read_csv(current_csv)
    merged = pd.merge(df_base, df_adv, on='sample_id', suffixes=('_base', '_adv'))

    # 统计损失减少的样本
    decreased = (merged['loss_adv'] < merged['loss_base']).sum()
    total = len(merged)

    if mode == "targeted":
        return int(decreased)  # 有目标攻击：损失减少算成功
    else:
        return int(total - decreased)


def run_blackbox_automation():
    base_save_dir = CFG.TRAIN.CKPT_SAVE_DIR
    summary_file = "blackbox_etth_dl_pt_targeted_freq_removal_summary.csv"

    if os.path.exists(summary_file):
        all_results = pd.read_csv(summary_file).to_dict('records')
    else:
        all_results = []

    # --- 步骤 0: 检查/生成干净样本的 Baseline ---
    if not os.path.exists(CLEAN_BASELINE_CSV):
        print("🚩 未发现干净 Baseline，正在生成...")
        cfg_base = EasyDict(deepcopy(CFG))
        cfg_base.DATASET.PARAM.use_npz = False  # 确保读原始数据
        cfg_base['MODEL.NAME'], cfg_base['MD5'] = cfg_base.MODEL.NAME, "baseline_run"
        runner = cfg_base.RUNNER(cfg_base)
        runner.init_test(cfg_base)
        runner.load_model(ckpt_path=TARGET_MODEL_CKPT)
        runner.test_clean(save_results=True)  # 会生成 test_losses_per_sample.csv
        # 搬运文件
        os.rename(os.path.join(runner.ckpt_save_dir, "test_losses_per_sample.csv"), CLEAN_BASELINE_CSV)
        del runner
        torch.cuda.empty_cache()

    # --- 步骤 1: 遍历对抗来源跑迁移性测试 ---
    for src in BLACKBOX_SOURCES:
        is_finished = any(r['Source_NPZ'] == src['name'] for r in all_results)
        if is_finished:
            print(f"⏩ 跳过已完成黑盒测试: {src['name']}")
            continue

        exp_name = f"blackbox_from_{src['name']}"
        print(f"\n" + "=" * 70)
        print(f"🕵️ 正在进行黑盒迁移测试: 来源={src['name']} | 模式={src['mode']}")
        print("=" * 70)

        curr_cfg = EasyDict(deepcopy(CFG))
        # 🌟 触发 Dataset 注入逻辑
        curr_cfg.DATASET.PARAM.use_npz = True
        curr_cfg.DATASET.PARAM.npz_path = src['path']

        curr_cfg['MODEL.NAME'] = curr_cfg.MODEL.NAME
        curr_cfg['MD5'] = exp_name
        curr_cfg.TRAIN.CKPT_SAVE_DIR = base_save_dir

        try:
            runner = curr_cfg.RUNNER(curr_cfg)
            runner.init_test(curr_cfg)
            runner.load_model(ckpt_path=TARGET_MODEL_CKPT)
            runner.model.eval()
            # 运行黑盒评估 (test_pp)
            test_return = runner.test_pp(save_results=True, save_metrics=True)

            # 定位生成的 CSV
            gen_csv = os.path.join(runner.ckpt_save_dir, "test_losses_per_sample.csv")

            # 计算成功率
            success_num = get_success_count(CLEAN_BASELINE_CSV, gen_csv, src['mode'])
            total_num = len(pd.read_csv(gen_csv))

            # 整理结果
            m = test_return.get("metrics_summary", {})
            entry = {
                "Source_NPZ": src['name'],
                "Attack_Mode": src['mode'],
                "MAE": m.get("MAE", 0),
                "MSE": m.get("MSE", 0),
                "WAPE": m.get("WAPE", 0),
                "Success_Samples": success_num,
                "ASR": f"{(success_num / total_num):.2%}",
                "Time_In": exp_name
            }
            all_results.append(entry)
            pd.DataFrame(all_results).to_csv(summary_file, index=False)
            print(f"✅ 记录完成: ASR={entry['ASR']}, MAE={entry['MAE']:.4f}")

        except Exception as e:
            print(f"❌ 黑盒测试失败: {src['name']}")
            print(f"原因: {str(e)}")

        finally:
            if 'runner' in locals(): del runner
            torch.cuda.empty_cache()

    print(f"\n✨ 黑盒迁移实验全部完成！汇总表: {summary_file}")


if __name__ == "__main__":
    run_blackbox_automation()
