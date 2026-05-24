#!/usr/bin/env python3
"""医学指南评测数据集生成脚本

从已上传的医学指南文档自动生成评测问题。

使用方法:
    python scripts/generate_eval_dataset.py                    # 生成25条
    python scripts/generate_eval_dataset.py --count 50         # 生成50条
    python scripts/generate_eval_dataset.py --output custom.jsonl  # 自定义输出
"""

import argparse
import json
import random
from pathlib import Path

# 预设问题模板（按主题分类）
QUESTION_TEMPLATES = {
    "儿童肺炎支原体肺炎诊疗指南（2025年版）": [
        ("定义", "肺炎支原体肺炎（MPP）的定义是什么？"),
        ("诊断", "儿童肺炎支原体肺炎的诊断标准是什么？"),
        ("症状", "MPP的临床表现有哪些？"),
        ("传播", "MPP的传播途径是什么？"),
        ("检验", "MP病原学检查方法有哪些？"),
        ("耐药", "MP耐药的主要原因是什么？"),
        ("影像", "MPP的影像学表现有哪些？"),
        ("鉴别", "MPP需要与哪些疾病进行鉴别诊断？"),
        ("分型", "轻症和重症MPP如何区分？"),
        ("MUMPP", "大环内酯类药物无反应性肺炎支原体肺炎（MUMPP）的定义是什么？"),
        ("SMPP", "重症肺炎支原体肺炎（SMPP）有哪些并发症？"),
        ("PB", "如何早期识别塑形性支气管炎（PB）？"),
        ("肺栓塞", "MPP出现肺栓塞时D-二聚体的诊断阈值是多少？"),
        ("坏死性肺炎", "坏死性肺炎（NP）的诊断依据是什么？"),
        ("治疗", "大环内酯类药物治疗MPP的疗程是多久？"),
    ],
    "肥胖症诊疗指南（2024年版）": [
        ("定义", "肥胖症的定义是什么？"),
        ("诊断", "如何诊断肥胖症？"),
        ("治疗", "肥胖症的治疗原则是什么？"),
        ("饮食", "肥胖症患者的饮食管理原则是什么？"),
        ("运动", "肥胖症患者如何进行运动治疗？"),
        ("药物", "肥胖症的药物治疗指征是什么？"),
        ("手术", "减重手术的适应症有哪些？"),
    ],
    "胃癌诊疗指南（2022年版）": [
        ("定义", "胃癌的早期症状有哪些？"),
        ("诊断", "胃癌的诊断方法有哪些？"),
        ("分期", "胃癌如何进行分期？"),
        ("治疗", "早期胃癌的治疗原则是什么？"),
        ("手术", "胃癌的手术方式有哪些？"),
    ],
    "膀胱癌诊疗指南（2022年版）": [
        ("危险因素", "膀胱癌的危险因素有哪些？"),
        ("症状", "膀胱癌的典型症状是什么？"),
        ("诊断", "膀胱癌的诊断方法有哪些？"),
        ("分级", "膀胱癌的组织学分级是什么？"),
        ("治疗", "非肌层浸润性膀胱癌的治疗方法有哪些？"),
    ],
    "宫颈癌诊疗指南（2022年版）": [
        ("筛查", "宫颈癌的筛查方法有哪些？"),
        ("诊断", "宫颈癌的诊断流程是什么？"),
        ("分期", "宫颈癌的临床分期如何判定？"),
        ("治疗", "早期宫颈癌的治疗方法有哪些？"),
    ],
    "乳腺癌诊疗指南（2022年版）": [
        ("体征", "乳腺癌的典型体征是什么？"),
        ("诊断", "乳腺癌的诊断方法有哪些？"),
        ("分子分型", "乳腺癌的分子分型有哪些？"),
        ("治疗", "乳腺癌的综合治疗原则是什么？"),
    ],
    "淋巴瘤诊疗指南（2022年版）": [
        ("分类", "淋巴瘤的分类有哪些？"),
        ("诊断", "淋巴瘤的诊断要点是什么？"),
        ("分期", "淋巴瘤如何进行分期？"),
        ("治疗", "侵袭性淋巴瘤的治疗原则是什么？"),
    ],
    "甲状腺癌诊疗指南（2022版）": [
        ("病理", "甲状腺癌的病理类型最常见的是哪种？"),
        ("诊断", "甲状腺癌的诊断方法有哪些？"),
        ("治疗", "乳头状甲状腺癌的治疗原则是什么？"),
        ("随访", "甲状腺癌术后如何进行随访？"),
    ],
    "前列腺癌诊疗指南（2022年版）": [
        ("诊断", "前列腺癌的诊断方法有哪些？"),
        ("评分", "Gleason评分系统是什么？"),
        ("治疗", "局限性前列腺癌的治疗选择有哪些？"),
    ],
    "原发性肺癌诊疗指南（2022年版）": [
        ("类型", "肺癌的主要组织学类型有哪些？"),
        ("诊断", "肺癌的诊断方法有哪些？"),
        ("分期", "非小细胞肺癌的分期如何判定？"),
        ("治疗", "晚期NSCLC的治疗原则是什么？"),
    ],
    "原发性肝癌诊疗指南（2024年版）": [
        ("诊断", "肝癌的影像学诊断方法有哪些？"),
        ("分期", "肝癌的临床分期如何判定？"),
        ("治疗", "早期肝癌的治疗方法有哪些？"),
        ("介入", "肝动脉化疗栓塞术（TACE）的适应症是什么？"),
    ],
    "流行性感冒诊疗方案（2025版）": [
        ("抗病毒", "流行性感冒的抗病毒治疗药物有哪些？"),
        ("区别", "流感和普通感冒的主要区别是什么？"),
        ("诊断", "流感的诊断标准是什么？"),
        ("预防", "流感疫苗的接种对象有哪些？"),
    ],
}

# 难度标签
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]

# 安全敏感标签
SAFETY_SENSITIVE_QUERIES = {
    "诊断标准", "治疗原则", "药物", "手术", "适应症", "禁忌症",
    "MUMPP", "SMPP", "肺栓塞", "坏死性", "抗病毒"
}


def load_documents(doc_dir: Path) -> dict[str, str]:
    """加载医学指南文档"""
    docs = {}
    if doc_dir.exists():
        for md_file in doc_dir.glob("*.md"):
            docs[md_file.stem] = md_file.read_text(encoding="utf-8")
    return docs


def generate_answer(query: str, doc_name: str, docs: dict[str, str]) -> str:
    """根据查询和文档生成参考答案（简化版）"""
    # 实际应用中应该用 LLM 来生成
    # 这里返回占位符
    return f"关于{doc_name}中{query}的详细答案请参考原文。"


def generate_dataset(count: int = 25, output: str = None) -> list[dict]:
    """生成评测数据集"""
    dataset = []

    # 展平所有问题
    all_questions = []
    for doc_name, questions in QUESTION_TEMPLATES.items():
        for topic, question in questions:
            all_questions.append((doc_name, topic, question))

    # 随机采样
    if count > len(all_questions):
        # 如果数量不够，重复采样
        sampled = all_questions * (count // len(all_questions) + 1)
        random.shuffle(sampled)
        sampled = sampled[:count]
    else:
        sampled = random.sample(all_questions, count)

    for doc_name, topic, question in sampled:
        # 判断是否安全敏感
        safety_sensitive = any(kw in question for kw in SAFETY_SENSITIVE_QUERIES)

        # 随机分配难度
        difficulty = random.choice(DIFFICULTY_LEVELS)

        entry = {
            "query": question,
            "expected_answer": "",  # 实际使用时需要人工标注
            "relevant_doc_ids": [doc_name],
            "difficulty": difficulty,
            "safety_sensitive": safety_sensitive,
        }
        dataset.append(entry)

    # 打乱顺序
    random.shuffle(dataset)

    return dataset


def main():
    parser = argparse.ArgumentParser(description="生成医学评测数据集")
    parser.add_argument("--count", type=int, default=25, help="生成数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    dataset = generate_dataset(count=args.count)

    output_path = Path(args.output) if args.output else Path("data/evaluation_dataset.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"已生成 {len(dataset)} 条评测数据，保存至: {output_path}")

    # 统计信息
    safety_count = sum(1 for e in dataset if e["safety_sensitive"])
    print(f"\n数据集统计:")
    print(f"  - 总数量: {len(dataset)}")
    print(f"  - 安全敏感问题: {safety_count} ({safety_count/len(dataset)*100:.1f}%)")
    print(f"  - 文档覆盖: {len(set(e['relevant_doc_ids'][0] for e in dataset))} 种")


if __name__ == "__main__":
    main()
