"""Command-line interface: python -m app.cli --help"""
import argparse
from app import core

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="retail-basket-prediction-analyzer",
        description="Compare an N-gram baseline with a small neural language model on synthetic retail baskets."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the starter project is wired up")
    run_parser = sub.add_parser("run", help="run all five project milestones")
    run_parser.add_argument("--baskets", type=int, default=5000)
    run_parser.add_argument("--epochs", type=int, default=80)
    pred_parser = sub.add_parser("predict", help="predict next products from a shopping sequence")
    pred_parser.add_argument("items", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Retail Basket Prediction Analyzer is ready.")
    elif args.command == "run":
        print("Milestone 1/5: generating synthetic baskets with planted itemsets...")
        result = core.run_pipeline(args.baskets, args.epochs)
        print(f"Generated {result['generated']['n_baskets']} baskets, {result['generated']['n_itemsets']} planted itemsets, {result['generated']['n_products']} products.")
        print("Milestone 2/5: trained N-gram baseline.")
        print(f"Milestone 3/5: pretrained neural language model ({result['neural']['training_examples']} next-item examples).")
        print("Milestone 4/5: evaluation complete.")
        for name in ("ngram", "neural_lm"):
            m = result["metrics"][name]
            print(f"{name:10s} MRR={m['mrr']:.4f} | Hits@1={m['hits_at_1']:.4f} | Hits@3={m['hits_at_3']:.4f} | Mean rank={m['mean_rank']:.2f}")
        print("Milestone 5/5: report and charts saved in outputs/.")
    elif args.command == "predict":
        items = args.items or input("Enter purchased items separated by commas: ").split(",")
        for rank, (item, score) in enumerate(core.predict_next(items), 1):
            print(f"{rank}. {item.replace('_', ' ')} ({score*100:.1f}%)")

if __name__ == "__main__":
    main()
