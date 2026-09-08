import json

VERSION = "2026-09-09.langgraph.6.paper_roles"

WRITEBACK_RULES_V1 = """\n【协商成果写回核对】
先逐项核对本次发言中提出、回应和确认的事项，再形成完整意见。区分已获双方明确接受的局部安排、尚未接受的单方提案、附条件的承诺，以及仍未解决的实体争点。
在overall_opinion中如实保留本次双方已确认的具体事项及其条件，同时明确仍有分歧的事实、责任和请求；在相关reason中说明接受或保留的依据。使用实际发言内容，不以原先总体立场覆盖本次确实接受的事项。
程序安排达成一致不等于责任归属达成一致；实体争点未解决也不意味着任何局部协议都未形成。不能把双方已明确接受的安排重新概括成“仅有一方提议”或“一概未形成协议”。
若对话没有明确的相互接受，就如实写尚未达成；不要为了满足本要求创造协议。对方单方断言不是事实，共同同意未来核验也不代表核验结论已经成立。
Q1至Q10仍严格按原来的命题和二元口径填写，不要求因存在局部协议而翻转立场。只保存真实进展与真实分歧，不预设达成全面一致。"""


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def case_text(case):
    return dump({key: case[key] for key in ("title", "abstract", "facts", "claimant_position",
                                           "respondent_position", "contract_terms", "issues")})


REVIEW_RULES = """只输出JSON，包含agent_id（字符串）、overall_opinion（非空字符串）、issues（数组）。
issues必须恰好包含Q1至Q10各一次，每项包含id（整数）、stance（整数0或1）、confidence（0至1）、reason（理由）。
二元口径：Q1至Q7表示对当前命题是否认可，认可为1，否则为0；不能把未来鉴定可能成立当作当前事实已经成立。
Q8与Q9表示当前是否接受支持相应请求：完整或明确的部分支持为1，当前不支持为0；仅讨论未来条件方案但尚未承诺支持时为0，并在reason说明条件。
Q10表示当前是否认可船厂整体胜诉，不用其他Q项机械推算。总体意见必须与各Q项一致。
每个reason写明当前理由与仍保留的条件。可以保持分歧，不要为了提高共识改写事实。"""


def initial(case, agent):
    perspective = ("角色规定分析职责和视角，不规定程序强制的胜负答案；中立技术鉴定人不属于甲乙任何一方。"
                   if case.get("role_profile") == "paper_appendix_v1" else
                   "阵营规定分析视角，不规定程序强制的胜负答案。")
    return (
        "你是海事争议多智能体系统的一名代理。根据指定角色独立分析，只使用给定材料。" + REVIEW_RULES,
        f"代理身份：{agent['id']}\n角色：{agent['role_prompt']}\n案件：{case_text(case)}\n"
        "形成你的初始完整意见。" + perspective,
    )


def summary(case, agent_id, review):
    return (
        "你负责将法律意见整理为忠实摘要。不要改变立场或消除条件。只输出JSON，包含summary字符串。",
        f"争点定义：{dump(case['issues'])}\n待摘要意见：{dump(review)}\n"
        "保留事实与因果判断、合同解释、责任分配、救济和最终结论、证据缺口及条件性安排。不要添加新的共同立场。",
    )


def similarity(first, second):
    return (
        "你是比较同一案件两份意见的中立语义评分员。只根据两份意见评分。",
        f"意见A：\n{first}\n\n意见B：\n{second}\n\n"
        "比较事实认定、法律推理、责任分配和救济结论的语义共识。0表示完全对立，0.5表示部分一致，1表示高度一致。"
        "共同案件背景和相同格式不算观点一致；共同同意鉴定不代表因果事实已一致。"
        "只输出一个score XML标签，标签内填写你判断的0至1两位小数，不输出解释。",
    )


def bilateral(case, agent, opponent, reviews, history, prior_interactions, *, mode="baseline"):
    system, user = (
        "你是参加双边协商的高级法律顾问。寻求有事实和理由支持的解决方案。",
        f"代理：{agent['id']}；角色：{agent['role_prompt']}\n案件：{case_text(case)}\n"
        f"你在本次互动前的意见：{dump(reviews[agent['id']])}\n对方{opponent}的意见：{dump(reviews[opponent])}\n"
        f"此前本分支双边记录：{dump(prior_interactions)}\n本次交替发言：{dump(history)}\n"
        "回应对方至少一个具体争点，明确维持、接受或提出的条件性让步；区分已接受方案与新提议。"
        "仅输出本次新发言，尽量不超过200个汉字。",
    )
    if mode == "appendix_consensus_v1":
        system = ("你是代表指定一方参加多轮双边协商的高级法律顾问。你的首要目标是共同解决争议、"
                  "寻求双方可接受的方案，而非赢得辩论或坚持原先立场。仍须忠实于给定事实和证据。")
        user += ("\n【附录中的协商任务】承认并回应对方新提出的具体顾虑；表现合作意愿，"
                 "把对话引向解决方案；提出能够缩小分歧的建设性步骤或折衷。只输出对对方的本次回应。")
    elif mode != "baseline":
        raise ValueError("Unknown bilateral prompt mode")
    if agent["faction"] == "neutral":
        system = system.replace("代表指定一方参加多轮双边协商的高级法律顾问", "保持独立中立、参加多轮双边协商的技术鉴定人")
        system = system.replace("参加双边协商的高级法律顾问", "参加双边协商的中立技术鉴定人")
        system += "你是中立技术鉴定人，不代表甲乙任何一方；协商不改变你的中立身份，不得将待检验假设写成已有检测结果。"
    return system, user


def revise(case, agent, previous, evidence, stage, *, writeback_mode="baseline"):
    system, user = (
        "你负责在交互后更新自己的完整意见。" + REVIEW_RULES,
        f"更新类型：{stage}\n代理：{agent['id']}；角色：{agent['role_prompt']}\n案件：{case_text(case)}\n"
        f"原完整意见：{dump(previous)}\n本次可见材料：{dump(evidence)}\n"
        "只将你确实接受的新理由、让步或证据写入当前立场；未涉及的争点保留原立场。"
        "在对应reason解释变化依据。调解和双边结果不是强制命令；拒绝或部分接受时说明理由。"
        "旁观者只能吸收双边双方确实共同接受且自己认可的内容，不能将单方提议视为协议。",
    )
    if writeback_mode == "agreement_aware_v1":
        user += WRITEBACK_RULES_V1
    elif writeback_mode != "baseline":
        raise ValueError("Unknown pair writeback mode")
    return system, user


def multilateral(case, agent, reviews, metrics, transcript):
    return (
        "你是一名参加多边讨论的海事争议代理。所有判断限于所提供案件。",
        f"代理：{agent['id']}；角色：{agent['role_prompt']}\n案件：{case_text(case)}\n"
        f"本轮开始时的五份意见：{dump(reviews)}\n本轮分歧统计：{dump(metrics)}\n"
        f"本轮此前发言：{dump(transcript)}\n"
        "针对一个主要分歧提出回应，说明可以接受的观点、仍坚持的理由及可能的交换方案。只输出不超过300字的发言。",
    )


def mediation(case, reviews, transcript, metrics):
    return (
        "你是中立的中央调解者。提出候选方案，由五名代理独立判断。只输出JSON。",
        f"案件：{case_text(case)}\n当前意见：{dump(reviews)}\n分歧统计：{dump(metrics)}\n发言：{dump(transcript)}\n"
        "输出diagnosis（分歧诊断）、proposed_agreements（候选条款数组）、unresolved（未解决问题数组）。"
        "不能按多数强制Q10胜负，不能把调解方案写成已达成协议。",
    )


def determination(case, mechanism, reviews):
    identities = ("五名成员身份：" +
                  dump([{key: a[key] for key in ("id", "name", "faction")} for a in case["agents"]]) + "\n"
                  if case.get("role_profile") == "paper_appendix_v1" else "")
    return (
        "你负责整理实验中的最终建议，保留未解决分歧。不读取参考裁决。只输出JSON。",
        f"案件：{case_text(case)}\n机制：{mechanism}\n{identities}最终五份意见：{dump(reviews)}\n"
        "输出disposition（非空字符串）、reasoning（非空字符串）、issue_findings（数组，Q1至Q10各一次，包含整数id和非空字符串finding）、"
        "unresolved_disagreements（字符串数组，无未决分歧时用空数组）。共识程度不能被当作法律正确性或真实仲裁效力。",
    )
