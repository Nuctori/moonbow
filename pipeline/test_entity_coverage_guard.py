# -*- coding: utf-8 -*-
"""pipeline/test_entity_coverage_guard.py

严谨回归与对抗压力测试套件：
针对两阶段特化微管线（287M GLiNER 语用定性 + 117M 客体向量对齐），验证三大核心缺口的彻底解决：
1. 复合需求早退防御（提 A+B 助手只做 A 时，被严格保持挂起）
2. 复合需求完整闭合（提 A+B 助手做完 A 和 B 时，成功核销）
3. 方案放弃与撤销（DROPPED / ABANDON 时安全出栈，拒绝死锁）
4. 增量推进与多待办精准选择性撤销回归
"""
from __future__ import annotations

import os
import sys
import time
import torch

# 确保导入路径包含项目根目录
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from pipeline.entity_coverage_guard import EntityCoverageGuard


def run_tests():
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print("\n" + "=" * 90)
    print(f"  两阶段特化微管线客体覆盖守卫 (Entity Coverage Guard) 回归与压力测试")
    print(f"  计算设备: {device} | 运行环境: .venv_xpu Python")
    print("=" * 90)

    guard = EntityCoverageGuard(device=device)

    total_tests = 0
    passed_tests = 0
    test_records = []

    def record_result(suite_name: str, case_name: str, passed: bool, detail: str, lat_ms: float):
        nonlocal total_tests, passed_tests
        total_tests += 1
        if passed:
            passed_tests += 1
        test_records.append({
            "suite": suite_name,
            "case": case_name,
            "passed": passed,
            "detail": detail,
            "lat_ms": lat_ms,
        })
        mark = "✓ PASS" if passed else "✗ FAIL"
        print(f"[{mark}] [{suite_name}] {case_name} ({lat_ms:.1f}ms)")
        print(f"       详情: {detail}")

    # =========================================================================
    # 测试集 1: 复合需求早退防御（提 A+B，助手只做 A -> 必须保持挂起，拒绝早退放行）
    # =========================================================================
    print("\n" + "-" * 75)
    print("【测试集 1: 复合需求早退漏洞防御 (Multi-Intent Premature Resolution)】")
    print("-" * 75)

    cases_suite1 = [
        {
            "name": "端口修改(9090) + JWT中间件 -> 仅做端口修改",
            "req": "把服务的端口修改为 9090，并且新增 JWT 鉴权校验中间件",
            "res": "已在 server.py 中将端口成功修改为 9090。",
            "expected_status": "PARTIAL",
            "expected_missing": ["JWT"],
            "expected_block": True,
        },
        {
            "name": "拼写修复(utils.py) + 依赖升级与测试跑通 -> 仅修拼写",
            "req": "修复 utils.py 里的拼写错误，顺便把依赖升级到最新版并跑通测试",
            "res": "已修复 utils.py 里的拼写错误。",
            "expected_status": "PARTIAL",
            "expected_missing": ["依赖升级到最新版并跑通测试"],
            "expected_block": True,
        },
        {
            "name": "增改接口(user.py) + 权限过滤(admin.py) -> 仅增接口",
            "req": "在 user.py 中增加 getUserList 接口，并且在 admin.py 中添加权限过滤",
            "res": "已经在 user.py 中新增了 getUserList 接口实现。",
            "expected_status": "PARTIAL",
            "expected_missing": ["admin.py"],
            "expected_block": True,
        },
        {
            "name": "Redis缓存 + MySQL连接池配置 -> 仅集成Redis",
            "req": "集成 Redis 缓存模块，并将 MySQL 连接池大小设置为 20",
            "res": "已成功集成 Redis 缓存模块并完成基础联调。",
            "expected_status": "PARTIAL",
            "expected_missing": ["MySQL"],
            "expected_block": True,
        },
    ]

    for tc in cases_suite1:
        # 重置单测状态
        guard.open_questions.clear()
        guard.events.clear()
        guard.issue_seq = 0

        t0 = time.time()
        ev_u = guard.step("user", tc["req"])
        ev_a = guard.step("assistant", tc["res"])
        chk = guard.check_return()
        lat = (time.time() - t0) * 1000.0

        is_blocked = chk["block"] == tc["expected_block"]
        has_q = len(guard.open_questions) == 1
        status_ok = has_q and guard.open_questions[0]["status"] == tc["expected_status"]
        missing_ok = has_q and any(m in guard.open_questions[0]["missing_entities"] for m in tc["expected_missing"])

        passed = is_blocked and status_ok and missing_ok
        diag = (
            f"抽取实体: {guard.all_questions.get('q001', {}).get('entities')} | "
            f"助手状态: {ev_a['action']} | "
            f"门禁拦截: {chk['block']} (未闭合待办数: {chk['open_count']})"
        )
        record_result("复合需求早退", tc["name"], passed, diag, lat)

    # =========================================================================
    # 测试集 2: 复合需求完整闭合（提 A+B，助手全部完成 -> 成功核销放行）
    # =========================================================================
    print("\n" + "-" * 75)
    print("【测试集 2: 复合需求完整核销 (Multi-Intent Full Resolution)】")
    print("-" * 75)

    cases_suite2 = [
        {
            "name": "单轮完成 端口修改 + JWT中间件",
            "req": "把服务的端口修改为 9090，并且新增 JWT 鉴权校验中间件",
            "res": "已在 server.py 中将端口修改为 9090，并且完成了 JWT 鉴权校验中间件的接入。",
            "expected_block": False,
        },
        {
            "name": "单轮完成 拼写修复 + 依赖升级与测试通过",
            "req": "修复 utils.py 里的拼写错误，顺便把依赖升级到最新版并跑通测试",
            "res": "已修复 utils.py 里的拼写错误，并且已经全部完成依赖升级，自动化测试全部跑通通过。",
            "expected_block": False,
        },
        {
            "name": "多轮增量完成 (Turn 1 只做 A -> PARTIAL, Turn 2 做完 B -> CLOSED)",
            "req": "在 user.py 中增加 getUserList 接口，并且在 admin.py 中添加权限过滤",
            "res_turn1": "已经在 user.py 中新增了 getUserList 接口实现。",
            "res_turn2": "已在 admin.py 中配置并实现了权限过滤逻辑，全部修改完成并通过验证。",
            "expected_block": False,
        }
    ]

    for tc in cases_suite2:
        guard.open_questions.clear()
        guard.events.clear()
        guard.issue_seq = 0

        t0 = time.time()
        guard.step("user", tc["req"])
        if "res_turn2" in tc:
            # 多轮增量测试
            ev1 = guard.step("assistant", tc["res_turn1"])
            chk_mid = guard.check_return()
            mid_ok = chk_mid["block"] and guard.open_questions[0]["status"] == "PARTIAL"
            ev2 = guard.step("assistant", tc["res_turn2"])
            chk_final = guard.check_return()
            passed = mid_ok and (chk_final["block"] == tc["expected_block"]) and (len(guard.open_questions) == 0)
            lat = (time.time() - t0) * 1000.0
            diag = f"Turn1状态: {ev1['action']} (阻断={chk_mid['block']}) -> Turn2状态: {ev2['action']} (放行={not chk_final['block']})"
            record_result("复合完整核销", tc["name"], passed, diag, lat)
        else:
            ev = guard.step("assistant", tc["res"])
            chk = guard.check_return()
            lat = (time.time() - t0) * 1000.0
            passed = (chk["block"] == tc["expected_block"]) and (len(guard.open_questions) == 0)
            diag = f"核销动作: {ev['action']} | 门禁放行: {not chk['block']} | 剩余待办: {len(guard.open_questions)}"
            record_result("复合完整核销", tc["name"], passed, diag, lat)

    # =========================================================================
    # 测试集 3: 方案放弃与撤销（DROPPED / ABANDON -> 安全出栈，解决死锁）
    # =========================================================================
    print("\n" + "-" * 75)
    print("【测试集 3: 方案放弃与撤销 (DROPPED / ABANDON Safe Unstacking)】")
    print("-" * 75)

    cases_suite3 = [
        {
            "name": "PARTIAL 挂起状态下放弃剩余需求 (做了一半，另一半不要了)",
            "steps": [
                ("user", "把服务的端口修改为 9090，并且新增 JWT 鉴权校验中间件"),
                ("assistant", "已在 server.py 中将端口成功修改为 9090。"),
                ("user", "刚才那个 JWT 先不要做了，放弃吧"),
            ],
            "expect_final_block": False,
            "expect_status": "DROPPED",
        },
        {
            "name": "直接撤销未开始的复杂任务 (算了，撤销任务)",
            "steps": [
                ("user", "重构 auth 模块并编写 test_login 单元测试"),
                ("user", "算了，auth 模块暂时不动了，撤销该任务不要了"),
            ],
            "expect_final_block": False,
            "expect_status": "DROPPED",
        },
        {
            "name": "多待办并存时的精准选择性撤销 (精准出栈目标待办，保留其他待办)",
            "steps": [
                ("user", "把服务的端口修改为 9090"),
                ("user", "集成 Redis 缓存模块"),
                ("user", "Redis 缓存先不做了，取消这个需求"),
            ],
            "expect_final_block": True,  # 端口任务仍在，应保持阻断
            "expect_remaining_q": "把服务的端口修改为 9090",
        },
    ]

    for tc in cases_suite3:
        guard.open_questions.clear()
        guard.events.clear()
        guard.issue_seq = 0

        t0 = time.time()
        for role, text in tc["steps"]:
            guard.step(role, text)
        chk = guard.check_return()
        lat = (time.time() - t0) * 1000.0

        if "expect_status" in tc:
            passed = (chk["block"] == tc["expect_final_block"]) and (guard.all_questions["q001"]["status"] == tc["expect_status"])
            diag = f"最终门禁阻断: {chk['block']} | 终态: {guard.all_questions['q001']['status']} (安全移出待办表)"
        else:
            # 选择性撤销
            has_one = len(guard.open_questions) == 1
            rem_match = has_one and (guard.open_questions[0]["text"] == tc["expect_remaining_q"])
            passed = (chk["block"] == tc["expect_final_block"]) and rem_match
            diag = f"精准保留指定待办: {[q['text'] for q in guard.open_questions]} | 阻断={chk['block']}"

        record_result("放弃撤销解死锁", tc["name"], passed, diag, lat)

    # =========================================================================
    # 测试集 4: 跨话题负例与拒绝执行防御
    # =========================================================================
    print("\n" + "-" * 75)
    print("【测试集 4: 跨话题冒充与拒绝执行拦截 (Antonym & Refusal Guard)】")
    print("-" * 75)

    cases_suite4 = [
        {
            "name": "助手明确无法执行/拒绝升级",
            "req": "升级 Python 到 3.12",
            "res": "经评估，Python 3.12 与现有依赖冲突严重，决定不升级，维持 3.11。",
            "expected_block": True,
        },
        {
            "name": "跨话题假完成拦截 (问端口，回答测试报告)",
            "req": "把服务的端口修改为 9090",
            "res": "单元测试已全部执行完毕，耗时 12.3 秒，覆盖率 94%。",
            "expected_block": True,
        }
    ]

    for tc in cases_suite4:
        guard.open_questions.clear()
        guard.events.clear()
        guard.issue_seq = 0

        t0 = time.time()
        guard.step("user", tc["req"])
        ev_a = guard.step("assistant", tc["res"])
        chk = guard.check_return()
        lat = (time.time() - t0) * 1000.0

        passed = chk["block"] == tc["expected_block"]
        diag = f"助手动作: {ev_a['action']} | 拦截诊断: {ev_a['detail'][:60]}... | 阻断={chk['block']}"
        record_result("跨话题与拒绝拦截", tc["name"], passed, diag, lat)

    # =========================================================================
    # 评测结果与汇总
    # =========================================================================
    print("\n" + "=" * 90)
    print(f"  测试汇总: 共执行 {total_tests} 项测试 | 通过: {passed_tests} 项 | 失败: {total_tests - passed_tests} 项")
    print(f"  测试通过率: {passed_tests / total_tests:.1%}")
    print("=" * 90)

    avg_lat = sum(r["lat_ms"] for r in test_records) / len(test_records)
    print(f"平均单场景端到端耗时: {avg_lat:.1f} ms")

    if passed_tests == total_tests:
        print("\n>>> ALL TESTS PASSED: 所有对抗缺陷已全部通过严格工程闭环验证！<<<")
        return 0
    else:
        print("\n>>> TEST FAILED: 存在未通过用例，请检查上述诊断日志！<<<")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
