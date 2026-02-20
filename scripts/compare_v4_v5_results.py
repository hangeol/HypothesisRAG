import json
from pathlib import Path


def main() -> None:
    # Paths to the two result files
    v4_path = Path("/mnt/data1/home/hangeol/project/HypothesisRAG/results/medqa_planning_v4_full.json")
    v5_path = Path("/mnt/data1/home/hangeol/project/HypothesisRAG/results/medqa_planning_v5_full.json")

    with v4_path.open("r", encoding="utf-8") as f:
        v4_data = json.load(f)
    with v5_path.open("r", encoding="utf-8") as f:
        v5_data = json.load(f)

    v4_results = {item["question_id"]: item for item in v4_data.get("results", [])}
    v5_results = {item["question_id"]: item for item in v5_data.get("results", [])}

    common_ids = sorted(set(v4_results.keys()) & set(v5_results.keys()))

    # 전체 "V4 맞고 V5 틀린" 케이스
    total_v4_correct_v5_wrong = 0
    total_v4_correct_v5_wrong_ids = []

    # 그 중에서 best_guess를 유지한 경우 (틀렸는데 유지)
    best_guess_maintained_wrong_count = 0
    best_guess_maintained_wrong_ids = []

    # 전체 "V5 맞고 V4 틀린" 케이스
    total_v5_correct_v4_wrong = 0
    total_v5_correct_v4_wrong_ids = []

    # 그 중에서 best_guess를 유지한 경우 (맞았는데 유지)
    best_guess_maintained_correct_count = 0
    best_guess_maintained_correct_ids = []

    for qid in common_ids:
        v4_item = v4_results[qid]
        v5_item = v5_results[qid]

        correct_answer = v4_item.get("correct_answer")
        if correct_answer is None:
            continue

        v4_mode = v4_item.get("modes", {}).get("planning_v4", {})
        v5_mode = v5_item.get("modes", {}).get("planning_v5", {})

        v4_correct = v4_mode.get("is_correct")
        v5_correct = v5_mode.get("is_correct")

        # --- 케이스 1: V4에서는 맞고, V5에서는 틀린 경우 ---
        if v4_correct is True and v5_correct is False:
            total_v4_correct_v5_wrong += 1
            total_v4_correct_v5_wrong_ids.append(qid)

            v5_plan = v5_mode.get("plan", {}) or {}
            best_guess_v5 = v5_plan.get("best_guess")
            predicted_v5 = v5_mode.get("predicted_answer")

            # best_guess 가 정답이 아니고, 그대로 predicted_answer 로 사용된 경우만 센다.
            if (
                best_guess_v5 is not None
                and predicted_v5 is not None
                and best_guess_v5 == predicted_v5
                and best_guess_v5 != correct_answer
            ):
                best_guess_maintained_wrong_count += 1
                best_guess_maintained_wrong_ids.append(qid)

        # --- 케이스 2: V5에서는 맞고, V4에서는 틀린 경우 ---
        if v5_correct is True and v4_correct is False:
            total_v5_correct_v4_wrong += 1
            total_v5_correct_v4_wrong_ids.append(qid)

            v5_plan = v5_mode.get("plan", {}) or {}
            best_guess_v5 = v5_plan.get("best_guess")
            predicted_v5 = v5_mode.get("predicted_answer")

            # best_guess 를 그대로 predicted_answer 로 사용해서 맞춘 경우
            if (
                best_guess_v5 is not None
                and predicted_v5 is not None
                and best_guess_v5 == predicted_v5
            ):
                best_guess_maintained_correct_count += 1
                best_guess_maintained_correct_ids.append(qid)

    # ----- 출력: 케이스 1 (V4 맞고 V5 틀림) -----
    print("=" * 60)
    print("V4에서는 맞았는데 V5에서 틀린 경우:", total_v4_correct_v5_wrong, "개")
    print("=" * 60)
    print()
    print("그 중에서 best_guess를 유지하다가 틀린 경우:", best_guess_maintained_wrong_count, "개")

    if total_v4_correct_v5_wrong > 0:
        percentage = (best_guess_maintained_wrong_count / total_v4_correct_v5_wrong) * 100
        print(f"비율: {percentage:.2f}%")
    else:
        print("비율: 0% (분모가 0)")

    print()
    print("전체 V4 맞고 V5 틀린 question_id 목록:", total_v4_correct_v5_wrong_ids)
    print()
    print("best_guess를 유지한 question_id 목록:", best_guess_maintained_wrong_ids)

    # ----- 출력: 케이스 2 (V5 맞고 V4 틀림) -----
    print()
    print("=" * 60)
    print("V5에서는 맞았는데 V4에서 틀린 경우:", total_v5_correct_v4_wrong, "개")
    print("=" * 60)
    print()
    print("그 중에서 best_guess를 유지해서 맞춘 경우:", best_guess_maintained_correct_count, "개")

    if total_v5_correct_v4_wrong > 0:
        percentage2 = (best_guess_maintained_correct_count / total_v5_correct_v4_wrong) * 100
        print(f"비율: {percentage2:.2f}%")
    else:
        print("비율: 0% (분모가 0)")

    print()
    print("전체 V5 맞고 V4 틀린 question_id 목록:", total_v5_correct_v4_wrong_ids)
    print()
    print("best_guess를 유지한 question_id 목록:", best_guess_maintained_correct_ids)


if __name__ == "__main__":
    main()

