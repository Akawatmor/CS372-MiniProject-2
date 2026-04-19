from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_RUN_DIR = REPO_ROOT / "code-run"


def new_notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    nb_cells = []
    for cell_type, source in cells:
        if cell_type == "markdown":
            nb_cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))
        elif cell_type == "code":
            nb_cells.append(nbf.v4.new_code_cell(dedent(source).strip()))
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    nb.cells = nb_cells
    return nb


def write_notebook(path: Path, cells: list[tuple[str, str]]) -> None:
    nb = new_notebook(cells)
    nbf.write(nb, path)


def execute_notebook(path: Path, timeout: int = 1800) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    nbf.write(nb, path)


def build_q1_cells() -> list[tuple[str, str]]:
    return [
        (
            "markdown",
            """
            # Q1 - Search Space and Memory Explosion
            คำนวณจำนวน itemsets ที่เป็นไปได้สำหรับจำนวน items = 30, 100, 200
            """,
        ),
        (
            "code",
            """
            import pandas as pd

            items = [30, 100, 200]
            bytes_per_itemset = 100


            def sci(x: int) -> str:
                return f"{x:.3e}"


            rows = []
            for n in items:
                total_itemsets = (1 << n) - 1
                estimated_bytes = total_itemsets * bytes_per_itemset
                rows.append(
                    {
                        "items": n,
                        "total_itemsets_exact": total_itemsets,
                        "total_itemsets_sci": sci(float(total_itemsets)),
                        "estimated_memory_bytes_sci": sci(float(estimated_bytes)),
                    }
                )

            df = pd.DataFrame(rows)
            print("Itemset count table:")
            print(df.to_string(index=False))

            growth_30_to_100 = ((1 << 100) - 1) / ((1 << 30) - 1)
            growth_100_to_200 = ((1 << 200) - 1) / ((1 << 100) - 1)
            growth_30_to_200 = ((1 << 200) - 1) / ((1 << 30) - 1)

            print("\\nGrowth ratios:")
            print(f"30 -> 100 items : {growth_30_to_100:.3e}x")
            print(f"100 -> 200 items: {growth_100_to_200:.3e}x")
            print(f"30 -> 200 items : {growth_30_to_200:.3e}x")
            """,
        ),
    ]


def build_q2_cells() -> list[tuple[str, str]]:
    return [
        (
            "markdown",
            """
            # Q2 - Frequent Itemset (Apriori vs FP-Growth vs Compact)
            ครอบคลุมข้อ 2.1 และ 2.2 โดยใช้ข้อมูล transactions 1,000 / 3,000 / 5,000 / 10,000
            """,
        ),
        (
            "code",
            """
            import time
            import tracemalloc
            import warnings

            import pandas as pd
            from mlxtend.frequent_patterns import apriori, fpgrowth

            warnings.filterwarnings("ignore")
            pd.set_option("display.max_rows", 200)
            pd.set_option("display.width", 120)
            """,
        ),
        (
            "code",
            """
            urls = {
                1000: "https://raw.githubusercontent.com/pakornlee/ml_example/23665225ce5781e8ea782e18829e6108a5a4c92f/transactions_1000_onehot.csv",
                3000: "https://raw.githubusercontent.com/pakornlee/ml_example/23665225ce5781e8ea782e18829e6108a5a4c92f/transactions_3000_onehot.csv",
                5000: "https://raw.githubusercontent.com/pakornlee/ml_example/23665225ce5781e8ea782e18829e6108a5a4c92f/transactions_5000_onehot.csv",
                10000: "https://raw.githubusercontent.com/pakornlee/ml_example/23665225ce5781e8ea782e18829e6108a5a4c92f/transactions_10000_onehot.csv",
            }

            datasets = {}
            for size, url in urls.items():
                df = pd.read_csv(url)
                df.columns = df.columns.str.strip()
                datasets[size] = df.astype(bool)

            print("Loaded datasets:", list(datasets.keys()))
            print("Shape for N=1000:", datasets[1000].shape)
            """,
        ),
        (
            "code",
            """
            def measure(func):
                tracemalloc.start()
                start = time.time()
                result = func()
                elapsed = time.time() - start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                return result, elapsed, peak / 1024 / 1024


            def run_apriori(df, min_sup, max_len=8):
                return apriori(df, min_support=min_sup, use_colnames=True, max_len=max_len)


            def run_fp(df, min_sup, max_len=8):
                return fpgrowth(df, min_support=min_sup, use_colnames=True, max_len=max_len)


            def run_compact(df, min_sup, max_k=8, top_n=20):
                total = len(df)
                bit_data = {}

                for col in df.columns:
                    bitstring = "".join(df[col].astype(int).astype(str))
                    bit_data[frozenset([col])] = int(bitstring, 2)

                levels = {1: {}}
                for itemset, bit in bit_data.items():
                    support = bin(bit).count("1") / total
                    if support >= min_sup:
                        levels[1][itemset] = (bit, support)

                levels[1] = dict(
                    sorted(levels[1].items(), key=lambda x: x[1][1], reverse=True)[:top_n]
                )

                for k in range(2, max_k + 1):
                    levels[k] = {}
                    prev = list(levels[k - 1].keys())
                    candidates = set()

                    for i in range(len(prev)):
                        for j in range(i + 1, len(prev)):
                            union = prev[i] | prev[j]
                            if len(union) == k:
                                candidates.add(union)

                    for c in candidates:
                        if all((c - frozenset([x])) in levels[k - 1] for x in c):
                            items = list(c)
                            bit = bit_data[frozenset([items[0]])]
                            for item in items[1:]:
                                bit &= bit_data[frozenset([item])]
                            support = bin(bit).count("1") / total
                            if support >= min_sup:
                                levels[k][c] = (bit, support)

                    if len(levels[k]) == 0:
                        break

                    levels[k] = dict(
                        sorted(levels[k].items(), key=lambda x: x[1][1], reverse=True)[:top_n]
                    )

                results = []
                for k, level in levels.items():
                    for itemset, (_, sup) in level.items():
                        results.append((itemset, sup))

                return results


            def count_ge2_apriori(df_result):
                return int((df_result["itemsets"].apply(len) >= 2).sum())


            def count_ge2_compact(compact_result):
                return sum(1 for itemset, _ in compact_result if len(itemset) >= 2)
            """,
        ),
        (
            "code",
            """
            # 2.1 Algorithm comparison (N=1000, min_sup=0.05)
            min_sup = 0.05
            df = datasets[1000]

            ap_result, ap_time, ap_mem = measure(lambda: run_apriori(df, min_sup, max_len=6))
            fp_result, fp_time, fp_mem = measure(lambda: run_fp(df, min_sup, max_len=6))
            cp_result, cp_time, cp_mem = measure(lambda: run_compact(df, min_sup, max_k=6))

            table_21 = pd.DataFrame(
                [
                    ["Apriori", len(ap_result), ap_time, ap_mem],
                    ["FP-Growth", len(fp_result), fp_time, fp_mem],
                    ["Compact", len(cp_result), cp_time, cp_mem],
                ],
                columns=["algorithm", "frequent_itemsets_count", "time_sec", "peak_memory_mb"],
            )
            print("Q2.1 - Algorithm Comparison")
            print(table_21.to_string(index=False))
            """,
        ),
        (
            "code",
            """
            # 2.2.1 Scaling by number of transactions (min_sup=0.05)
            rows = []
            for size in [1000, 3000, 5000, 10000]:
                df = datasets[size]

                ap_res, ap_t, ap_m = measure(lambda: run_apriori(df, 0.05, max_len=6))
                fp_res, fp_t, fp_m = measure(lambda: run_fp(df, 0.05, max_len=6))
                cp_res, cp_t, cp_m = measure(lambda: run_compact(df, 0.05, max_k=6))

                rows.extend(
                    [
                        [size, "Apriori", len(ap_res), ap_t, ap_m],
                        [size, "FP-Growth", len(fp_res), fp_t, fp_m],
                        [size, "Compact", len(cp_res), cp_t, cp_m],
                    ]
                )

            table_221 = pd.DataFrame(
                rows,
                columns=["transactions", "algorithm", "frequent_itemsets_count", "time_sec", "peak_memory_mb"],
            )
            print("Q2.2.1 - Scaling by transaction count")
            print(table_221.to_string(index=False))
            """,
        ),
        (
            "code",
            """
            # 2.2.2 Vary min_support at N=1000 (count only itemsets with size >= 2)
            rows = []
            df = datasets[1000]

            for ms in [0.01, 0.05, 0.1, 0.2]:
                ap_res, ap_t, ap_m = measure(lambda: run_apriori(df, ms, max_len=6))
                fp_res, fp_t, fp_m = measure(lambda: run_fp(df, ms, max_len=6))
                cp_res, cp_t, cp_m = measure(lambda: run_compact(df, ms, max_k=6))

                rows.extend(
                    [
                        [ms, "Apriori", count_ge2_apriori(ap_res), ap_t, ap_m],
                        [ms, "FP-Growth", count_ge2_apriori(fp_res), fp_t, fp_m],
                        [ms, "Compact", count_ge2_compact(cp_res), cp_t, cp_m],
                    ]
                )

            table_222 = pd.DataFrame(
                rows,
                columns=["min_support", "algorithm", "count_ge_2_itemsets", "time_sec", "peak_memory_mb"],
            )
            print("Q2.2.2 - Min support variation (>=2-itemset)")
            print(table_222.to_string(index=False))
            """,
        ),
        (
            "code",
            """
            # 2.2.3 Vary max_itemset k at N=1000, min_sup=0.05 (count only >=2-itemset)
            rows = []
            df = datasets[1000]

            for k in [4, 5, 6, 7, 8]:
                ap_res, ap_t, ap_m = measure(lambda: run_apriori(df, 0.05, max_len=k))
                fp_res, fp_t, fp_m = measure(lambda: run_fp(df, 0.05, max_len=k))
                cp_res, cp_t, cp_m = measure(lambda: run_compact(df, 0.05, max_k=k))

                rows.extend(
                    [
                        [k, "Apriori", count_ge2_apriori(ap_res), ap_t, ap_m],
                        [k, "FP-Growth", count_ge2_apriori(fp_res), fp_t, fp_m],
                        [k, "Compact", count_ge2_compact(cp_res), cp_t, cp_m],
                    ]
                )

            table_223 = pd.DataFrame(
                rows,
                columns=["max_k", "algorithm", "count_ge_2_itemsets", "time_sec", "peak_memory_mb"],
            )
            print("Q2.2.3 - Max itemset (k) variation (>=2-itemset)")
            print(table_223.to_string(index=False))
            """,
        ),
    ]


def build_q3_cells() -> list[tuple[str, str]]:
    return [
        (
            "markdown",
            """
            # Q3 - Association Rule Filtering (A/B/C)
            A: support + confidence, B: + lift > 1, C: + p-value < 0.05
            """,
        ),
        (
            "code",
            """
            import warnings
            from itertools import combinations

            import pandas as pd
            from scipy.stats import chi2_contingency

            warnings.filterwarnings("ignore")
            pd.set_option("display.max_rows", 200)
            pd.set_option("display.width", 140)
            """,
        ),
        (
            "code",
            """
            url = "https://raw.githubusercontent.com/pakornlee/ml_example/23665225ce5781e8ea782e18829e6108a5a4c92f/transactions_1000_onehot.csv"
            df = pd.read_csv(url)
            df.columns = df.columns.str.strip()
            df = df.astype(bool)

            min_sup = 0.05
            min_conf = 0.3
            top_n = 30
            print("Dataset shape:", df.shape)
            """,
        ),
        (
            "code",
            """
            def run_compact(dataframe, min_support, max_k=6, top_n_keep=30):
                total = len(dataframe)
                bit_data = {}
                for col in dataframe.columns:
                    bitstring = "".join(dataframe[col].astype(int).astype(str))
                    bit_data[frozenset([col])] = int(bitstring, 2)

                levels = {1: {}}
                for itemset, bit in bit_data.items():
                    support = bin(bit).count("1") / total
                    if support >= min_support:
                        levels[1][itemset] = (bit, support)

                levels[1] = dict(
                    sorted(levels[1].items(), key=lambda x: x[1][1], reverse=True)[:top_n_keep]
                )

                for k in range(2, max_k + 1):
                    levels[k] = {}
                    prev = list(levels[k - 1].keys())
                    candidates = set()

                    for i in range(len(prev)):
                        for j in range(i + 1, len(prev)):
                            union = prev[i] | prev[j]
                            if len(union) == k:
                                candidates.add(union)

                    for c in candidates:
                        if all((c - frozenset([x])) in levels[k - 1] for x in c):
                            items = list(c)
                            bit = bit_data[frozenset([items[0]])]
                            for item in items[1:]:
                                bit &= bit_data[frozenset([item])]
                            support = bin(bit).count("1") / total
                            if support >= min_support:
                                levels[k][c] = (bit, support)

                    if len(levels[k]) == 0:
                        break

                    levels[k] = dict(
                        sorted(levels[k].items(), key=lambda x: x[1][1], reverse=True)[:top_n_keep]
                    )

                out = {}
                for level in levels.values():
                    for itemset, (_, support) in level.items():
                        out[itemset] = support
                return out


            def calc_p_value(dataframe, antecedent, consequent):
                x = dataframe[list(antecedent)].all(axis=1)
                y = dataframe[list(consequent)].all(axis=1)
                n11 = int(((x) & (y)).sum())
                n10 = int(((x) & (~y)).sum())
                n01 = int(((~x) & (y)).sum())
                n00 = int(((~x) & (~y)).sum())

                row1 = n11 + n10
                row2 = n01 + n00
                col1 = n11 + n01
                col2 = n10 + n00
                if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
                    return 1.0

                _, p_value, _, _ = chi2_contingency([[n11, n10], [n01, n00]])
                return float(p_value)


            def generate_rules(freq_itemsets, dataframe):
                rules = []
                for itemset, support_xy in freq_itemsets.items():
                    if len(itemset) < 2:
                        continue

                    for i in range(1, len(itemset)):
                        for antecedent_tuple in combinations(itemset, i):
                            antecedent = frozenset(antecedent_tuple)
                            consequent = itemset - antecedent

                            support_x = freq_itemsets.get(antecedent, 0.0)
                            support_y = freq_itemsets.get(consequent, 0.0)
                            if support_x == 0.0 or support_y == 0.0:
                                continue

                            confidence = support_xy / support_x
                            lift = confidence / support_y
                            p_value = calc_p_value(dataframe, antecedent, consequent)

                            rules.append(
                                {
                                    "antecedent": antecedent,
                                    "consequent": consequent,
                                    "support": support_xy,
                                    "confidence": confidence,
                                    "lift": lift,
                                    "p_value": p_value,
                                }
                            )

                return pd.DataFrame(rules)
            """,
        ),
        (
            "code",
            """
            freq_itemsets = run_compact(df, min_sup, max_k=6, top_n_keep=top_n)
            rules = generate_rules(freq_itemsets, df)

            rules_A = rules[(rules["support"] >= min_sup) & (rules["confidence"] >= min_conf)].copy()
            rules_B = rules_A[rules_A["lift"] > 1].copy()
            rules_C = rules_B[rules_B["p_value"] < 0.05].copy()

            removed_A = rules[(rules["support"] < min_sup) | (rules["confidence"] < min_conf)].copy()
            removed_B = rules_A[rules_A["lift"] <= 1].copy()
            removed_C = rules_B[rules_B["p_value"] >= 0.05].copy()

            print("All rules:", len(rules))
            print("(A) Basic rules:", len(rules_A))
            print("(B) Strong rules:", len(rules_B))
            print("(C) Significant rules:", len(rules_C))

            print("\\nExample removed in A:")
            if not removed_A.empty:
                print(removed_A.sort_values(["confidence", "support"]).head(1).to_string(index=False))
            else:
                print("No rule removed in A from this generated rule set.")

            print("\\nExample removed in B:")
            if not removed_B.empty:
                print(removed_B.sort_values("lift").head(1).to_string(index=False))
            else:
                print("No rule removed in B.")

            print("\\nExample removed in C:")
            if not removed_C.empty:
                print(removed_C.sort_values("p_value", ascending=False).head(1).to_string(index=False))
            else:
                print("No rule removed in C.")

            print("\\nTop 10 significant rules (C):")
            print(
                rules_C.sort_values(["lift", "confidence"], ascending=False)
                .head(10)
                .to_string(index=False)
            )
            """,
        ),
    ]


def build_q4_cells() -> list[tuple[str, str]]:
    return [
        (
            "markdown",
            """
            # Q4 - Real-world Dataset 1 (bread_basket.csv)
            ทดลองหลายค่า support/confidence และวิเคราะห์ Spurious + Surprising patterns
            """,
        ),
        (
            "code",
            """
            import warnings

            import pandas as pd
            from mlxtend.frequent_patterns import apriori, association_rules
            from scipy.stats import chi2_contingency

            warnings.filterwarnings("ignore")
            pd.set_option("display.max_rows", 200)
            pd.set_option("display.width", 150)
            """,
        ),
        (
            "code",
            """
            raw = pd.read_csv("dataset/bread_basket.csv")
            print("Raw rows:", len(raw))
            print("Unique transactions:", raw["Transaction"].nunique())
            print("Unique items:", raw["Item"].nunique())
            print("\\nTop 15 items:")
            print(raw["Item"].value_counts().head(15))

            basket = (
                raw.groupby(["Transaction", "Item"])["Item"]
                .count()
                .unstack(fill_value=0)
                .gt(0)
            )
            print("\\nBasket shape:", basket.shape)
            """,
        ),
        (
            "code",
            """
            support_grid = [0.01, 0.02, 0.03, 0.05]
            support_scan_rows = []
            for ms in support_grid:
                fi = apriori(basket, min_support=ms, use_colnames=True, max_len=5)
                support_scan_rows.append([ms, len(fi), int((fi["itemsets"].apply(len) >= 2).sum())])

            support_scan = pd.DataFrame(
                support_scan_rows,
                columns=["min_support", "all_frequent_itemsets", "frequent_itemsets_ge2"],
            )
            print("Support scan:")
            print(support_scan.to_string(index=False))
            """,
        ),
        (
            "code",
            """
            min_sup = 0.02
            min_conf = 0.30

            fi = apriori(basket, min_support=min_sup, use_colnames=True, max_len=5)
            rules = association_rules(
                fi,
                metric="support",
                min_threshold=min_sup,
                num_itemsets=len(fi),
            )

            rules_A = rules[(rules["support"] >= min_sup) & (rules["confidence"] >= min_conf)].copy()
            rules_B = rules_A[rules_A["lift"] > 1].copy()

            def p_value_for_rule(row):
                antecedent = list(row["antecedents"])
                consequent = list(row["consequents"])
                x = basket[antecedent].all(axis=1)
                y = basket[consequent].all(axis=1)
                n11 = int(((x) & (y)).sum())
                n10 = int(((x) & (~y)).sum())
                n01 = int(((~x) & (y)).sum())
                n00 = int(((~x) & (~y)).sum())

                row1 = n11 + n10
                row2 = n01 + n00
                col1 = n11 + n01
                col2 = n10 + n00
                if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
                    return 1.0

                _, p_value, _, _ = chi2_contingency([[n11, n10], [n01, n00]])
                return float(p_value)

            if not rules_B.empty:
                rules_B["p_value"] = rules_B.apply(p_value_for_rule, axis=1)
            else:
                rules_B["p_value"] = pd.Series(dtype=float)

            rules_C = rules_B[rules_B["p_value"] < 0.05].copy()
            removed_B = rules_A[rules_A["lift"] <= 1].copy()
            removed_C = rules_B[rules_B["p_value"] >= 0.05].copy()

            print(f"Chosen min_support={min_sup}, min_confidence={min_conf}")
            print("All rules:", len(rules))
            print("(A) Basic rules:", len(rules_A))
            print("(B) Strong rules:", len(rules_B))
            print("(C) Significant rules:", len(rules_C))

            print("\\nTop significant rules:")
            if not rules_C.empty:
                cols = ["antecedents", "consequents", "support", "confidence", "lift", "p_value"]
                print(rules_C.sort_values("lift", ascending=False).head(15)[cols].to_string(index=False))
            else:
                print("No significant rules.")

            print("\\nExamples of rules removed in B (lift <= 1):")
            if not removed_B.empty:
                print(removed_B[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))
            else:
                print("No rules removed in B.")

            print("\\nExamples of rules removed in C (p-value >= 0.05):")
            if not removed_C.empty:
                print(removed_C[["antecedents", "consequents", "support", "confidence", "lift", "p_value"]].head(10).to_string(index=False))
            else:
                print("No rules removed in C.")
            """,
        ),
    ]


def build_q5_cells() -> list[tuple[str, str]]:
    return [
        (
            "markdown",
            """
            # Q5 - Real-world Dataset 2 (spotify_dataset.csv)
            วิเคราะห์แบบแยกจาก bread_basket โดยใช้ user เป็น transaction และ artist เป็น item
            """,
        ),
        (
            "code",
            """
            import warnings

            import pandas as pd
            from mlxtend.frequent_patterns import apriori, association_rules
            from scipy.stats import chi2_contingency

            warnings.filterwarnings("ignore")
            pd.set_option("display.max_rows", 200)
            pd.set_option("display.width", 170)
            """,
        ),
        (
            "code",
            """
            # The full file is very large; this notebook intentionally uses a large sample window.
            sample_rows = 200_000
            raw = pd.read_csv("dataset/spotify_dataset.csv", on_bad_lines="skip", nrows=sample_rows)
            raw.columns = raw.columns.str.strip().str.replace('"', "", regex=False)

            for col in raw.columns:
                if raw[col].dtype == "object":
                    raw[col] = raw[col].str.replace('"', "", regex=False).str.strip()

            print("Loaded rows:", len(raw))
            print("Unique users:", raw["user_id"].nunique())
            print("Unique artists:", raw["artistname"].nunique())
            print("\\nTop 15 artists:")
            print(raw["artistname"].value_counts().head(15))
            """,
        ),
        (
            "code",
            """
            top_n_artists = 30
            top_artists = raw["artistname"].value_counts().head(top_n_artists).index
            filtered = raw[raw["artistname"].isin(top_artists)].copy()

            basket = (
                filtered.groupby(["user_id", "artistname"])["artistname"]
                .count()
                .unstack(fill_value=0)
                .gt(0)
            )

            print("Filtered rows (top 30 artists):", len(filtered))
            print("Basket shape:", basket.shape)

            support_grid = [0.05, 0.10, 0.15, 0.20, 0.25]
            support_scan_rows = []
            for ms in support_grid:
                fi = apriori(basket, min_support=ms, use_colnames=True, max_len=3)
                support_scan_rows.append([ms, len(fi), int((fi["itemsets"].apply(len) >= 2).sum())])

            support_scan = pd.DataFrame(
                support_scan_rows,
                columns=["min_support", "all_frequent_itemsets", "frequent_itemsets_ge2"],
            )
            print("\\nSupport scan:")
            print(support_scan.to_string(index=False))
            """,
        ),
        (
            "code",
            """
            min_sup = 0.15
            min_conf = 0.40

            fi = apriori(basket, min_support=min_sup, use_colnames=True, max_len=3)
            rules = association_rules(
                fi,
                metric="support",
                min_threshold=min_sup,
                num_itemsets=len(fi),
            )

            rules_A = rules[(rules["support"] >= min_sup) & (rules["confidence"] >= min_conf)].copy()
            rules_B = rules_A[rules_A["lift"] > 1].copy()

            def p_value_for_rule(row):
                antecedent = list(row["antecedents"])
                consequent = list(row["consequents"])
                x = basket[antecedent].all(axis=1)
                y = basket[consequent].all(axis=1)
                n11 = int(((x) & (y)).sum())
                n10 = int(((x) & (~y)).sum())
                n01 = int(((~x) & (y)).sum())
                n00 = int(((~x) & (~y)).sum())

                row1 = n11 + n10
                row2 = n01 + n00
                col1 = n11 + n01
                col2 = n10 + n00
                if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
                    return 1.0

                _, p_value, _, _ = chi2_contingency([[n11, n10], [n01, n00]])
                return float(p_value)

            if not rules_B.empty:
                rules_B["p_value"] = rules_B.apply(p_value_for_rule, axis=1)
            else:
                rules_B["p_value"] = pd.Series(dtype=float)

            rules_C = rules_B[rules_B["p_value"] < 0.05].copy()
            removed_B = rules_A[rules_A["lift"] <= 1].copy()
            removed_C = rules_B[rules_B["p_value"] >= 0.05].copy()

            print(f"Chosen min_support={min_sup}, min_confidence={min_conf}")
            print("All rules:", len(rules))
            print("(A) Basic rules:", len(rules_A))
            print("(B) Strong rules:", len(rules_B))
            print("(C) Significant rules:", len(rules_C))

            print("\\nTop significant rules:")
            if not rules_C.empty:
                cols = ["antecedents", "consequents", "support", "confidence", "lift", "p_value"]
                print(rules_C.sort_values("lift", ascending=False).head(20)[cols].to_string(index=False))
            else:
                print("No significant rules.")

            print("\\nRules removed in B (lift <= 1):", len(removed_B))
            if not removed_B.empty:
                print(removed_B[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).to_string(index=False))

            print("\\nRules removed in C (p-value >= 0.05):", len(removed_C))
            if not removed_C.empty:
                print(removed_C[["antecedents", "consequents", "support", "confidence", "lift", "p_value"]].head(10).to_string(index=False))
            """,
        ),
    ]


def main() -> None:
    CODE_RUN_DIR.mkdir(parents=True, exist_ok=True)

    notebooks = {
        "Q1_SearchSpace.ipynb": build_q1_cells(),
        "Q2_FrequentItemset.ipynb": build_q2_cells(),
        "Q3_AssociationRules.ipynb": build_q3_cells(),
        "Q4_BreadBasket.ipynb": build_q4_cells(),
        "Q5_Spotify.ipynb": build_q5_cells(),
    }

    print("Creating notebooks...")
    for name, cells in notebooks.items():
        path = CODE_RUN_DIR / name
        write_notebook(path, cells)
        print(f"  - created {path.relative_to(REPO_ROOT)}")

    print("\\nExecuting notebooks...")
    for name in notebooks:
        path = CODE_RUN_DIR / name
        print(f"  - executing {path.relative_to(REPO_ROOT)}")
        execute_notebook(path, timeout=2400)
        print(f"    done {name}")

    print("\\nAll notebooks created and executed successfully.")


if __name__ == "__main__":
    main()
