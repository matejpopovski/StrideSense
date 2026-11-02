# StrideSense

Human activity recognition for Android: an LSTM trained on UCI HAR, exported to LiteRT, with the
sensor preprocessing written once in Python and once in pure Kotlin and held together by tests. This
first commit holds the model and that shared pipeline. The Android app that uses them is built on
top of it.

The exported model scores **88.7%** over 2,947 windows from 9 people it never saw
(macro F1 0.886), ranging from 63.6% to 100.0% per person. The dataset was collected
with phones worn sideways at the waist, so these numbers do not describe a phone in a pocket, and
real-world accuracy is unmeasured.

## Layout

- `ml/`: training, LiteRT export, evaluation and the reference preprocessing.
- `core/`: the same preprocessing as a streaming Kotlin module with no Android dependency.
- `fixtures/parity/`: real windows that Python and Kotlin must agree on, to 2.7e-15.
- `app/src/main/assets/`: the exported model and the normalization it needs.
- `results/`: the output of every run, including per-subject accuracy.

## Run it

```bash
uv sync --locked
uv run pytest
./gradlew -p core ktlintCheck test --no-daemon
```

## Limitations

The model is trained on waist-mounted phones in a lab, knows six activities and has no "other"
class. Nothing here has run on a phone or an emulator.

## Data and licence

Code is MIT. The model weights and fixtures derive from Human Activity Recognition Using Smartphones
by Reyes-Ortiz, Anguita, Ghio, Oneto and Parra, from the
[UCI Machine Learning Repository](https://doi.org/10.24432/C54S4K), which is not committed here.
The UCI page lists CC BY 4.0, while the `README.txt` in the archive requires citing Anguita et al.,
ESANN 2013, and prohibits commercial use. This is a non-commercial project that credits both.
