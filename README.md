# AdaMus

AdaMus steers generated instrumental music with your concentration, measured
by EEG in real time: the music sounds different while you concentrate than
while you rest.

## Requirements

| | |
|---|---|
| Computer | Mac with Apple Silicon. The music model does not run in real time elsewhere. |
| System | macOS |
| Python | 3.12 |
| Headset | Muse 2 |
| To build the music generator | Xcode with its Metal toolchain, `cmake`, `git` and `curl`. |

## Installation

```sh
git clone <repository-url> AdaMus && cd AdaMus
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,generator]"
```

## Running

```sh
adamus
```

## Datasets

To run a session without a headset, the program plays a recording from the
`data/` folder in its place: `data/<dataset>/<recording>/` holds the samples of
every electrode (`eeg.csv`) and what the person was doing meanwhile
(`protocol.yaml`), which a session over the recording is judged against.

| Dataset | Source |
|---|---|
| `own_recordings` | Your own sessions with the headset: each one is stored here, to be played again. |
| `synthetic` | A computed signal, for the tests. |
| `wearable_monitoring_n_back` | People with a Muse 2 doing blocks of an n-back task with breaks, from [Regulation of Brain Cognitive States through Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring](https://physionet.org/content/brain-wearable-monitoring/1.0.0/) by Fekri Azgomi et al., under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/brain-wearable-monitoring/view-license/1.0.0/). The detector was trained on this dataset. |
| `office_mental_workload` | People with a Muse S watching a relaxing video and doing an n-back task, from [Measuring and Quantifying Mental Workload and Stress in Everyday Situations, focusing on typical office activities](https://doi.org/10.5281/zenodo.15681263) by Anders, Bhaduri and Arnrich, under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/). |

The recordings of other datasets are cut and resampled from the original files;
no electrode of the originals is left out.

## Configuration

Everything that can be tuned is in `config.yaml`.

## Tests

```sh
pytest                                                 # all tests
pytest -m "not realtime and not training"              # the quick tests
python scripts/mutation_testing.py                       # mutation testing
python scripts/mutation_testing.py src/adamus/config.py  # mutation testing of one file
```

## License

[MIT](LICENSE). The datasets in `data/` keep their own licenses, given above.
