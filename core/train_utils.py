from logging import Logger
from typing import Literal


# TODO - Implement
# def early_stop(
#     logger: Logger, early_stopping_monitor: Literal["val_loss", "train_loss"]
# ):
#     if early_stopping_monitor == "val_loss":
#         if not val_loader and not warning_printed:
#             warning_printed = True
#             logger.warning(
#                 "Early stopping is enabled but validation data loader is not provided."
#             )
#             return False

#         if len(self.history.val_loss) < 2:
#             return False

#         if self.config.early_stopping_min_delta_strategy == "fixed":
#             if (
#                 self.history.val_loss[-1]
#                 < self.history.val_loss[-2] + self.config.early_stopping_min_delta
#             ):
#                 early_stop_counter = 0
#             else:
#                 early_stop_counter += 1
#         elif self.config.early_stopping_min_delta_strategy == "previous_proportional":
#             if (
#                 self.history.val_loss[-1]
#                 < self.history.val_loss[-2]
#                 + self.history.val_loss[-2] * self.config.early_stopping_min_delta
#             ):
#                 early_stop_counter = 0
#             else:
#                 early_stop_counter += 1
#         elif self.config.early_stopping_min_delta_strategy == "delta_proportional":
#             if len(self.history.val_loss) >= 3:
#                 if (
#                     self.history.val_loss[-1]
#                     < self.history.val_loss[-2]
#                     + abs(self.history.val_loss[-3] - self.history.val_loss[-2])
#                     * self.config.early_stopping_min_delta
#                 ):
#                     early_stop_counter = 0
#                 else:
#                     early_stop_counter += 1
#     else:
#         if len(self.history.train_loss) < 2:
#             continue

#         if self.config.early_stopping_min_delta_strategy == "fixed":
#             if (
#                 self.history.train_loss[-1]
#                 < self.history.train_loss[-2] + self.config.early_stopping_min_delta
#             ):
#                 early_stop_counter = 0
#             else:
#                 early_stop_counter += 1
#         elif self.config.early_stopping_min_delta_strategy == "previous_proportional":
#             if (
#                 self.history.train_loss[-1]
#                 < self.history.train_loss[-2]
#                 + self.history.train_loss[-2] * self.config.early_stopping_min_delta
#             ):
#                 early_stop_counter = 0
#             else:
#                 early_stop_counter += 1
#         elif self.config.early_stopping_min_delta_strategy == "delta_proportional":
#             if len(self.history.train_loss) >= 3:
#                 if (
#                     self.history.train_loss[-1]
#                     < self.history.train_loss[-2]
#                     + abs(self.history.train_loss[-3] - self.history.train_loss[-2])
#                     * self.config.early_stopping_min_delta
#                 ):
#                     early_stop_counter = 0
#                 else:
#                     early_stop_counter += 1
