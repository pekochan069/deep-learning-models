from core.config import Config


# TODO - Implement
def early_stop(config: Config):
    early_stop_counter = 0

    def inner(history: list[float]):
        nonlocal early_stop_counter

        if len(history) < 2:
            return False

        if config.early_stopping_min_delta_strategy == "fixed":
            if history[-1] < history[-2] + config.early_stopping_min_delta:
                early_stop_counter = 0
            else:
                early_stop_counter += 1
        elif config.early_stopping_min_delta_strategy == "previous_proportional":
            if (
                history[-1]
                < history[-2] + history[-2] * config.early_stopping_min_delta
            ):
                early_stop_counter = 0
            else:
                early_stop_counter += 1
        elif config.early_stopping_min_delta_strategy == "delta_proportional":
            if len(history) >= 3:
                if (
                    history[-1]
                    < history[-2]
                    + abs(history[-3] - history[-2]) * config.early_stopping_min_delta
                ):
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
            else:
                return False

        if early_stop_counter == config.early_stopping_patience:
            return True

    return inner
