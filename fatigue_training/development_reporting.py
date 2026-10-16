"""Independent offline development report; never routes through legacy outer-OOF HTML."""
import csv
import html
import json
from pathlib import Path


def render(summary,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    escape=lambda value:html.escape(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    records=[];details=[]
    for key,result in sorted(summary.get("results",{}).items()):
        t=result["task"]
        records.append(dict(slot=key,model=t["model"],phase=t["phase"],fold=t["fold"],seed=t["seed"],lr=t["lr"],
            primary=result["primary"],fatigue_recall=result.get("fatigue_recall"),fatigue_supported_subjects=result.get("fatigue_supported_subjects"),scope="inner_validation"))
        details.append(f"<details><summary>{html.escape(key)} · 主指标 {result['primary']:.4f}</summary><pre>{escape(result)}</pre></details>")
    columns=["slot","model","phase","fold","seed","lr","primary","fatigue_recall","fatigue_supported_subjects","scope"]
    with (root/"development_summary.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(records)
    head={k:v for k,v in summary.items() if k not in ("results","comparisons")}
    page="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>开发内层报告</title>
<style>body{font:16px system-ui;margin:32px;max-width:1200px;color:#183040}pre{white-space:pre-wrap;background:#f2f5f7;padding:14px}summary{cursor:pointer;padding:10px}details{border-bottom:1px solid #ccd}</style>
<h1>CNN＋Mamba 开发内层报告</h1><p>范围：inner_validation。没有外层测试预测、formal协议完成或新独立确认。候选选择和同内层复用的乐观性仍在。</p>"""
    page+="<h2>状态、选定学习率、预算与阶段门</h2><pre>"+escape(head)+"</pre>"
    page+="<h2>按fit查看按人支持数、混淆与诊断</h2>"+"\n".join(details)
    page+="<h2>配对条件区间</h2><pre>"+escape(summary.get("comparisons",{}))+"</pre>"
    # Learning curves are textual, fully offline and linked only to this directory.
    curves=[]
    for p in sorted((root/"fits").glob("*/attempt*/learning_curve.json")):
        value=json.loads(p.read_text(encoding="utf-8"))
        curves.append(f"<details><summary>{html.escape(str(p.relative_to(root)))}</summary><pre>{escape(value)}</pre></details>")
    page+="<h2>学习曲线</h2>"+"\n".join(curves)+"</html>"
    (root/"development_report.html").write_text(page,encoding="utf-8")
    return root/"development_report.html"
