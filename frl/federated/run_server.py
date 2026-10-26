# Aggregation strategy for each privacy method (used by run_shared.py).

import os
import argparse
import math
import numpy as np
import types

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import flwr as fl
from frl.utils import parseConfig, require_run_id


def make_fit_config_fn(batches_per_agg: int, true_rounds: int):
    def fit_config_fn(server_round: int):
        true_round = math.ceil(server_round / batches_per_agg)
        return {
            'round'          : server_round,   # Flower sub-round (for logging/CSV)
            'true_round'     : true_round,     # true FL round (for epsilon schedule)
            'true_num_rounds': true_rounds,    # total true rounds (for epsilon denominator)
        }
    return fit_config_fn


def get_eval_fn():
    def evaluate(server_round, parameters, config):
        return 0.0, {'accuracy': 0.0}
    return evaluate


def build_strategy(cfg, **common_kwargs):
    """FL strategy for cfg['method'] (shared by run_server.py and run_shared.py)."""
    method  = cfg['method']
    epsilon = cfg['epsilon']
    lam     = cfg['lambda_smooth']
    if method in ('cpq', 'ldpfl'):
        # One-bit private quantization of the update (onebit.py, server_onebit.py)
        from frl.federated.server_onebit import OneBitStrategy
        strategy = OneBitStrategy(
            method       = method,
            epsilon      = epsilon,
            clip_radius  = cfg['clip_radius'],
            lambda_param = lam,
            pairing_granularity = cfg['cpq_pairing_granularity'],
            pairing_mode = cfg['cpq_pairing'],
            pairing_seed = cfg['seed'] * 1000003 + 17,
            **common_kwargs,
        )
        print(f'[SERVER] Strategy: OneBitStrategy ({method})')

    elif method in ('gaussian', 'laplace', 'none'):
        # Standard FedAvg — float weights, no binary decode needed.
        # We use PqcFedAvg from server.py but with oracle disabled so it
        # never waits for .npz files and immediately uses gRPC weights.
        from frl.federated.server import PqcFedAvg
        strategy = PqcFedAvg(**common_kwargs)
        print(f'[SERVER] Strategy: PqcFedAvg (standard FedAvg, method={method})')

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: none | cpq | ldpfl | gaussian | laplace"
        )
    return strategy


class BatchAccumulatingStrategy(fl.server.strategy.Strategy):
    """
    Wraps any Flower strategy to implement sequential sub-round batching.

    Every `batches_per_agg` Flower sub-rounds = 1 true FL round.
    Sub-rounds 1 to (batches_per_agg-1): accumulate results, return the
    unchanged current global params so the next batch trains from the
    same baseline model.
    Sub-round batches_per_agg: aggregate all accumulated results with
    FedAvg and update the global model.
    """

    def __init__(self, inner_strategy, batches_per_agg: int, true_rounds: int):
        self.inner           = inner_strategy
        self.batches_per_agg = batches_per_agg
        self.true_rounds     = true_rounds
        self._accumulated_results  = []   # (client, FitRes) from all sub-rounds
        self._accumulated_failures = []
        self._current_params = None       # last true global model parameters

    def initialize_parameters(self, client_manager):
        params = self.inner.initialize_parameters(client_manager)
        self._current_params = params
        return params

    def configure_fit(self, server_round, parameters, client_manager):
        base_params = self._current_params if self._current_params is not None else parameters
        fit_configs = self.inner.configure_fit(server_round, base_params, client_manager)

        # Inject true_round into every client config regardless of which
        # inner strategy built it. The one-bit strategies build config manually
        # and never call on_fit_config_fn, so we override here.
        true_round = math.ceil(server_round / self.batches_per_agg)
        updated = []
        for client, fit_ins in fit_configs:
            new_config = dict(fit_ins.config)
            new_config['true_round']      = true_round
            new_config['true_num_rounds'] = self.true_rounds
            updated.append((client, fl.common.FitIns(fit_ins.parameters, new_config)))
        return updated

    def aggregate_fit(self, server_round, results, failures):
        self._accumulated_results.extend(results)
        self._accumulated_failures.extend(failures)

        is_true_agg = (server_round % self.batches_per_agg == 0)

        if is_true_agg:
            true_round = server_round // self.batches_per_agg
            print(f"\n[BATCH] True round {true_round}: aggregating "
                  f"{len(self._accumulated_results)} updates "
                  f"({self.batches_per_agg} batches × "
                  f"{len(self._accumulated_results)//self.batches_per_agg} clients)")
            agg_params, metrics = self.inner.aggregate_fit(
                server_round,
                self._accumulated_results,
                self._accumulated_failures,
            )
            self._accumulated_results  = []
            self._accumulated_failures = []
            if agg_params is not None:
                self._current_params = agg_params
            return agg_params, metrics
        else:
            sub = server_round % self.batches_per_agg
            print(f"\n[BATCH] Sub-round {sub}/{self.batches_per_agg}: "
                  f"accumulated {len(self._accumulated_results)} updates so far, "
                  f"holding global model")
            # Return unchanged global model — clients in next sub-round
            # train from the same starting point
            return self._current_params, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        # Trigger evaluation only on true aggregation rounds
        is_true_agg = (server_round % self.batches_per_agg == 0)
        if is_true_agg:
            return self.inner.configure_evaluate(server_round, parameters, client_manager)
        return []   # no evaluation between sub-rounds

    def aggregate_evaluate(self, server_round, results, failures):
        is_true_agg = (server_round % self.batches_per_agg == 0)
        if is_true_agg:
            return self.inner.aggregate_evaluate(server_round, results, failures)
        return None, {}

    def evaluate(self, server_round, parameters):
        return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified FLDQN Server')
    parser.add_argument('--config', type=str, default='configs/base.yaml')
    # --vehicle is not used by the server but parseConfig() reads args.vehicle safely
    args = parser.parse_args()
    args.vehicle = None

    cfg         = parseConfig(args)
    method      = cfg['method']
    epsilon     = cfg['epsilon']
    delta       = cfg['delta']
    lam         = cfg['lambda_smooth']
    n_clients   = cfg['num_clients']    # from sumo.num_clients — single source
    n_rounds    = cfg['num_rounds']
    true_rounds     = cfg['true_rounds']
    batches_per_agg = cfg['batches_per_agg']
    use_ns3     = cfg['use_ns3']

    RUN_ID = require_run_id()
    print(f'[SERVER] FL_RUN_ID  = {RUN_ID}')
    print(f'[SERVER] method     = {method}')
    print(f'[SERVER] epsilon    = {epsilon}')
    print(f'[SERVER] n_clients  = {n_clients}')
    print(f'[SERVER] n_rounds   = {n_rounds}')
    print(f'[SERVER] use_ns3    = {use_ns3}')

    from frl.utils import setup_eval_csv
    eval_csv_path = setup_eval_csv(cfg['out_dir'], method, RUN_ID)
    print(f'[SERVER] Eval CSV  = {eval_csv_path}')

    # common_kwargs = dict(
    #     fraction_fit      = 1.0,
    #     fraction_evaluate = 0.0,
    #     min_fit_clients      = n_clients,
    #     min_evaluate_clients = 0,
    #     min_available_clients= n_clients,
    #     on_fit_config_fn  = make_fit_config_fn(batches_per_agg, true_rounds),
    #     evaluate_fn       = get_eval_fn(),
    # )
    common_kwargs = dict(
        fraction_fit         = 1.0,
        fraction_evaluate    = 1.0,
        min_fit_clients      = n_clients,
        min_evaluate_clients = n_clients,
        min_available_clients= n_clients,
        on_fit_config_fn     = make_fit_config_fn(batches_per_agg, true_rounds),
        on_evaluate_config_fn= lambda server_round: {
            'round': server_round,
            'eval_episodes': cfg.get('eval_episodes', 3),
        },
        evaluate_fn          = get_eval_fn(),
    )

    strategy = build_strategy(cfg, **common_kwargs)

    # Wrap any strategy with batch accumulation if batches_per_agg > 1
    if batches_per_agg > 1:
        strategy = BatchAccumulatingStrategy(strategy, batches_per_agg, true_rounds)
        print(f'[SERVER] Batch accumulation ENABLED: '
              f'{batches_per_agg} sub-rounds × {n_clients} clients = '
              f'{batches_per_agg * n_clients} updates per true round')
    else:
        print('[SERVER] Batch accumulation DISABLED (batches_per_agg=1)')


    from frl.utils import save_data

    from frl.env.shared_train import summary_row

    def _aggregate_evaluate(self, server_round, results, failures):
        sums = [eval_res.metrics for _, eval_res in results if 'tt_sum' in eval_res.metrics]
        if not sums:
            print(f"[SERVER] Round {server_round}: no eval results received")
            return None, {}
        row = summary_row(server_round, sums)
        print(f"[SERVER] Round {server_round} GLOBAL EVAL | travel mean={row[1]:.1f}s "
              f"std={row[2]:.1f}s | arrived={row[3]:.0%} | time loss={row[4]:.1f}s | n={row[6]}")
        save_data(eval_csv_path, row)
        return row[1], {'travel_time_mean': row[1], 'arrival_rate': row[3]}

    strategy.aggregate_evaluate = types.MethodType(_aggregate_evaluate, strategy)

    fl.server.start_server(
        server_address      = '0.0.0.0:8080',
        config              = fl.server.ServerConfig(num_rounds=n_rounds),
        grpc_max_message_length = 256 * 1024 * 1024,
        strategy            = strategy,
    )