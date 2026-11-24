# AdaMus

AdaMus steers generated instrumental music with your concentration, measured by EEG in real
time: the music sounds different while you concentrate than while you rest.

A Muse headset measures the EEG. Every half second the focus detector judges from the last
seconds of signal how likely it is that you concentrate, and the music generator (Magenta
RealTime 2 in Pure Data) moves the music between a quiet solo piano and a full piano concerto
to match. A session follows a protocol of blocks of rest and focus, so that how well the music
followed can be measured. Each session is stored with everything it recorded, and the app
shows its results and plays it back on a timeline.

## Datasets

A session can also run without a headset: a recording from the `data/` folder is played in its
place. `data/<dataset>/<recording>/` holds the samples of every electrode (`eeg.csv`) and what
the person was doing meanwhile (`protocol.yaml`), which the session is judged against.

| Dataset | Source |
|---|---|
| `own_recordings` | Your own sessions with the headset, each stored to be played again. They are the biodata of the people who wore it, so they are never put into git. |
| `synthetic` | A computed signal, for the tests. |
| `wearable_monitoring_n_back` | People with a Muse 2 doing blocks of an n-back task with breaks, from [Regulation of Brain Cognitive States through Auditory, Gustatory, and Olfactory Stimulation with Wearable Monitoring](https://physionet.org/content/brain-wearable-monitoring/1.0.0/) by Fekri Azgomi et al., under the [Open Data Commons Attribution License v1.0](https://physionet.org/content/brain-wearable-monitoring/view-license/1.0.0/). The focus detector is trained on this dataset. |
| `office_mental_workload` | People with a Muse S watching a relaxing video and doing an n-back task, from [Measuring and Quantifying Mental Workload and Stress in Everyday Situations, focusing on typical office activities](https://doi.org/10.5281/zenodo.15681263) by Anders, Bhaduri and Arnrich, under the [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/). |

The recordings are cut and resampled from the original files; no electrode is left out.

## Configuration

Everything that can be tuned is in `config.yaml`: the headset, the protocol, the music and the
timeouts. The sessions are stored in `sessions/`; what the program installs and trains itself
is in `models/`.

## License

[MIT](LICENSE). The datasets in `data/` keep their own licenses, given above.

## Requirements

| | |
|---|---|
| Computer | A Mac with Apple Silicon and macOS 26 or later: the music model runs in real time only there. |
| Python | 3.12 |
| Tools | Xcode 26 or later with its Metal toolchain, to build the app and the music generator; `cmake`, `git` and `curl` for the generator too. |
| Internet | On the first start only, to download what the program sets up. |
| Headset | A Muse 2, or none: then a recording of a dataset is played. |

## Running

Once, in a terminal:

```sh
git clone https://github.com/RastislavLiptak/AdaMus.git
cd AdaMus
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,generator]"
sh scripts/build_app.sh
```

Then open `AdaMus.app` in the project folder. It starts the program of the folder it lies in, so
anywhere else, such as in Applications, put an alias of it, not the app itself. 

## Tests

```sh
.venv/bin/pytest
.venv/bin/pytest -m fast
.venv/bin/pytest -m slow
.venv/bin/pytest -m window
.venv/bin/python scripts/mutation_testing.py
```

Every test; those of a minute; those of minutes; those that use the window with the mouse and
the keyboard (leave the Mac alone meanwhile); the mutation tests, which take hours. The window
tests need Automation Mode, allowed once by an administrator:

```sh
sudo automationmodetool enable-automationmode-without-authentication
```
