from __future__ import annotations

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fedmed.train import class_weights_from_counts, evaluate, initial_weights

from .task import label_counts, load_test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    rc = context.run_config
    rounds = int(rc["num-server-rounds"])
    data_path = str(rc["data-path"])
    n_nodes = int(context.run_config.get("num-partitions", 5))
    cw = class_weights_from_counts(label_counts(data_path, n_nodes, float(rc["alpha"])))

    if rc["strategy"] == "fedprox":
        strategy = FedProx(fraction_train=1.0, fraction_evaluate=1.0, proximal_mu=float(rc["proximal-mu"]))
    else:
        strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    test = load_test(data_path)

    def central_eval(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        ev = evaluate(arrays.to_numpy_ndarrays()[0], test)
        print(f"[flower] round {server_round}: test acc={ev['accuracy']:.3f} bal_acc={ev['balanced_accuracy']:.3f}")
        return MetricRecord({"accuracy": ev["accuracy"], "balanced_accuracy": ev["balanced_accuracy"]})

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord([initial_weights(0)]),
        train_config=ConfigRecord({"class-weights": cw}),
        num_rounds=rounds,
        evaluate_fn=central_eval,
    )
    np.save("flower_global_model.npy", result.arrays.to_numpy_ndarrays()[0])
