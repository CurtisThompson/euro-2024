from src.load_data import process_results
from src.build_model import build_model
from src.predict_tournament import predict_tournament


def run(sims=1000, verbose=True, tune_model=False, tuning_evals=100,
        no_randomness=False):
    """Run entire pipeline to train model and predict winners of tournament.
    
    :param sims: Number of simulations to run, defaults to 1000
    :param verbose: Whether to output pipeline logs, defaults to True
    :param tune_model: Whether to perform hyperparameter tuning, defaults to False
    :param tuning_evals: Maximum number of models to evaluate, defaults to 100
    :param no_randomness: Whether to remove added randomness, defaults to False

    :type sims: int
    :type verbose: bool
    :type tune_model: bool
    :type tuning_evals: int
    :type no_randomness: bool
    """
    # ETL
    process_results(verbose=verbose)

    # Build model
    build_model(verbose=verbose,
                tune_model=tune_model,
                tuning_evals=tuning_evals)

    # Simulate tournament
    predict_tournament(sims=sims,
                       verbose=verbose,
                       no_randomness=no_randomness)


if __name__ == '__main__':
    run(sims=1000, verbose=True)
