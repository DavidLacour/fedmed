from __future__ import annotations

import warnings

import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedmed.train import ClientState, DPConfig, LocalConfig, evaluate, local_train

from .task import load_partition

warnings.filterwarnings("ignore")
app = ClientApp()


def _data(context: Context):
    rc, nc = context.run_config, context.node_config
    return load_partition(str(rc["data-path"]), int(nc["partition-id"]), int(nc["num-partitions"]),
                          float(rc["alpha"]))


@app.train()
def train(msg: Message, context: Context) -> Message:
    w = msg.content["arrays"].to_numpy_ndarrays()[0]
    cfg = msg.content["config"]
    rc = context.run_config
    mu = float(cfg.get("proximal-mu", 0.0))
    sigma = float(rc["noise-multiplier"])
    lcfg = LocalConfig(
        algorithm="fedprox" if mu > 0 else "fedavg", mu=mu, lr=float(rc["lr"]),
        local_steps=int(rc["local-steps"]), class_weights=list(cfg["class-weights"]),
        dp=DPConfig(sigma) if sigma > 0 else None,
    )
    data = _data(context)
    rnd = int(cfg.get("server-round", 1))
    res = local_train(w, data, lcfg, ClientState(), seed=rnd * 31 + int(context.node_config["partition-id"]))
    metrics = {"train_loss": res.train_loss, "num-examples": res.n_samples}
    if lcfg.dp is not None:
        # ClientApps are stateless between rounds, so account for all rounds so far from scratch:
        # every round ran res.steps Poisson-sampled steps at the same sample rate and noise.
        from opacus.accountants import RDPAccountant

        acc = RDPAccountant()
        acc.history = [(sigma, lcfg.batch_size / len(data), res.steps * rnd)]
        metrics["epsilon"] = float(acc.get_epsilon(lcfg.dp.delta))
    content = RecordDict({"arrays": ArrayRecord([np.asarray(w + res.delta, np.float32)]),
                          "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    w = msg.content["arrays"].to_numpy_ndarrays()[0]
    ev = evaluate(w, _data(context))
    m = MetricRecord({"accuracy": ev["accuracy"], "balanced_accuracy": ev["balanced_accuracy"],
                      "num-examples": ev["n"]})
    return Message(content=RecordDict({"metrics": m}), reply_to=msg)
