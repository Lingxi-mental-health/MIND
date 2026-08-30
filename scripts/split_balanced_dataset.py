import json
import os
from collections import defaultdict

# 路径根目录：默认相对本仓库，可用环境变量覆盖，避免把内部基础设施路径写进代码
PROJECT_ROOT = os.environ.get("MIND_PROJECT_ROOT", ".")



def split_balanced(
    input_path: str,
    output_path_part1: str,
    output_path_part2: str,
    per_category_per_file: int = 25,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Use diagnosis as the 4 categories: Anxiety, Depression, Mix, Others
    target_categories = ["Anxiety", "Depression", "Mix", "Others"]
    items_by_category = defaultdict(list)
    for item in data:
        diag = item.get("diagnosis")
        items_by_category[diag].append(item)

    # Validate there are enough items per target category (should be 50 each)
    for c in target_categories:
        count = len(items_by_category.get(c, []))
        if count < per_category_per_file * 2:
            raise ValueError(
                f"Diagnosis '{c}' has {count} items; need at least {per_category_per_file * 2}."
            )

    # Deterministic split: first N per category -> part1; next N -> part2
    part1 = []
    part2 = []
    for c in target_categories:
        items = items_by_category[c]
        part1.extend(items[:per_category_per_file])
        part2.extend(items[per_category_per_file : per_category_per_file * 2])

    # Write outputs
    os.makedirs(os.path.dirname(output_path_part1), exist_ok=True)
    with open(output_path_part1, "w", encoding="utf-8") as f:
        json.dump(part1, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(output_path_part2), exist_ok=True)
    with open(output_path_part2, "w", encoding="utf-8") as f:
        json.dump(part2, f, ensure_ascii=False, indent=2)

    # Print summary
    def counts(items):
        counter = defaultdict(int)
        for it in items:
            counter[it.get("diagnosis")] += 1
        return dict(counter)

    print("Categories:", target_categories)
    print("Part1 counts:", counts(part1), "Total:", len(part1))
    print("Part2 counts:", counts(part2), "Total:", len(part2))


if __name__ == "__main__":
    base_dir = PROJECT_ROOT
    input_file = os.path.join(
        base_dir,
        "data/balanced_dataset_4categories_50each_natural_qwen32b.json",
    )
    out1 = os.path.join(
        base_dir,
        "data/balanced_dataset_4categories_50each_natural_qwen32b_part1_100.json",
    )
    out2 = os.path.join(
        base_dir,
        "data/balanced_dataset_4categories_50each_natural_qwen32b_part2_100.json",
    )

    split_balanced(input_file, out1, out2, per_category_per_file=25)


