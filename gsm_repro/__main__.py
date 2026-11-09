from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import config
from .common import atomic_json
from .engine import create_run,execute,load_run
from .manifests import prepare
from .reporting import export_report
from .scoring import score_all,failed_scores,SCORING_VERSION

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline-first GSM8K reproduction core; default backend is fake.")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare",help="Verify author/data hashes and write immutable manifests; no model calls")
    p.add_argument("--data",type=Path,default=ROOT.parent/"llm-multiagent_debate_needData")
    p.add_argument("--author",type=Path,default=ROOT.parent/"llm_multiagent_debate")
    p = sub.add_parser("plan",help="Freeze a local run profile without reading credentials or generating")
    p.add_argument("--config",type=Path,default=ROOT/"configs/development.fake.json")
    p.add_argument("--manifest",type=Path,default=ROOT/"manifests/train_debug_10.json")
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--limit",type=int)
    for name in ("run","resume"):
        p = sub.add_parser(name,help="Execute only pending tasks from an already planned profile")
        p.add_argument("--run-dir",type=Path,required=True)
        p.add_argument("--max-tasks",type=int)
        p.add_argument("--allow-model-calls",action="store_true")
        p.add_argument("--allow-test-evaluation",action="store_true")
    for name in ("report","status"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir",type=Path,required=True)
    p = sub.add_parser("score",help="Rescore saved outputs offline; retain original run scores")
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        value = prepare(args.data,args.author,ROOT)
    elif args.command == "plan":
        value = create_run(ROOT,args.run_dir,config.load(args.config),args.manifest,limit=args.limit)
    elif args.command in ("run","resume"):
        if args.max_tasks is not None and args.max_tasks < 1:
            raise ValueError("max-tasks must be positive")
        value = execute(ROOT,args.run_dir,allow_model_calls=args.allow_model_calls,
                        allow_test_evaluation=args.allow_test_evaluation,max_tasks=args.max_tasks)
        export_report(ROOT,args.run_dir)
    elif args.command == "report":
        value = export_report(ROOT,args.run_dir)
    elif args.command == "status":
        profile,manifest,store = load_run(ROOT,args.run_dir,executing=False)
        value = dict(profile_sha256=profile["profile_sha256"],tasks=store.tasks())
    else:
        values = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
        value = [dict(qid=r["qid"],method=r["method"],scores=score_all(r["gold"],r["outputs"],ROOT/"author_reference/eval_gsm.py")
                      if r["status"] == "completed" else failed_scores()) for r in values]
        if args.output.exists():
            raise ValueError("Choose a new scoring output file; do not overwrite existing evidence")
        atomic_json(args.output,dict(scoring_source="current core; original result file unchanged",
                                    scoring_version=SCORING_VERSION,results=value))
        value = dict(output=str(args.output),samples=len(values),model_calls=0)
    print(json.dumps(value,ensure_ascii=True,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; pending tasks can resume. Unconfirmed requests are not replayed.")
        raise SystemExit(130)
    except (ValueError,RuntimeError) as exc:
        print(json.dumps(dict(error=str(exc)),ensure_ascii=True))
        raise SystemExit(2)
