# Parity fixtures

Each file in `parity/` holds a few windows of sensor events and the model windows that
`ml/preprocess.py` produces from them. `core/` replays the events through the Kotlin pipeline and
must reach the same result. Regenerate them with `python -m ml.make_fixtures`.

The signals are excerpts of held-out recordings from the UCI HAR dataset, resampled onto simulated
sensor clocks. Attribution and licence notes are in the "Data and licence" section of the root
README. Source: Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X. (2013). Human
Activity Recognition Using Smartphones. UCI Machine Learning Repository.
https://doi.org/10.24432/C54S4K
