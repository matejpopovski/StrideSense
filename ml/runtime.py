"""The Python LiteRT runtime, wrapped the way the app calls it: one window, one thread."""

from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter


class LiteRtModel:
    def __init__(self, path: Path) -> None:
        self.interpreter = Interpreter(model_path=str(path), num_threads=1)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input["index"], window[np.newaxis])
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output["index"])[0].copy()

    def predict(self, windows: np.ndarray) -> np.ndarray:
        return np.stack([self(window) for window in windows])

    def builtin_ops(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.interpreter._get_ops_details():
            counts[op["op_name"]] = counts.get(op["op_name"], 0) + 1
        return counts
