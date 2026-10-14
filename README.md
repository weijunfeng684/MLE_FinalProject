# Survival Is Not Enough: Three Paths to Value-Based Learning in Bomberman

Machine Learning Essentials final project, Heidelberg University, summer
semester 2026.

Team Hildebrandslied: Yimin Yan, Junfeng Wei, and Zhikai Zhang.

This project studies how value-based Bomberman agents can score while handling
delayed explosions and competing opponents. The three development branches use
different forms of structured knowledge: Siegfried constrains neural action
values with a tactical and safety controller, SEAT-Q adds temporal supervision
and staged policy adaptation to a compact representation, and DQFD_RE studies
demonstration-based learning after tabular and neural baselines.

## Models in the report

| Model | Main design | Retained policy or outcome |
| --- | --- | --- |
| Siegfried | Convolutional Dueling Double DQN with spatial and action-conditioned features, followed by task-specific tactical and safety control | Selected as the tournament agent |
| SEAT-Q | Compact Dueling Double DQN with multi-step learning, temporal hazard features, auxiliary escape supervision, and staged adaptation | League checkpoint at 225,000 stage-local transitions |
| DQFD_RE | Feature-based neural Q-learning with four-step returns, symmetry augmentation, and demonstration pre-training | Demonstrations retained for initialization; continued demonstration supervision did not improve the tested policy |

The models share a Q-learning foundation, but their representations, training
procedures, and deployed action-selection rules differ. Siegfried has a
deployment-time safety controller. SEAT-Q's planner supplies training signals
and does not create an equivalent inference shield. DQFD_RE leaves final action
selection to its learned policy.

## Evaluation

The course score is

```text
score = coins + 5 × opponent kills
```

Training return is kept separate from this score. The project also records
coins, kills, self-kills, deaths caused by opponents, survival, invalid actions,
and runtime failures where the evaluation logs provide them.

### Branch-specific results

- Siegfried's final-package Task 4 check used 300 development rounds. It scored
  1,598 points against 818, 835, and 834 for the three official rule-based
  agents. The run recorded a mean placement of 1.92, no timeout, no runtime
  error, and a maximum logged decision time of 0.010 seconds.
- SEAT-Q retained the League policy. In its separate 1,500-game evaluation, it
  scored 5.4700 per game against 5.2167 for the official rule-based opponent.
  Its additional coins offset its lower kill rate in that batch. A matched test
  did not establish a benefit for the later Recovery gate.
- DQFD_RE's development results improved after attack features and symmetry
  augmentation were added. Continued demonstration supervision did not improve
  on using demonstrations only for initialization under the tested settings.

These results come from different opponents, seeds, and selection procedures.
They should not be pooled into one cross-model score.

### Shared-game comparison

The report also compares the complete agents in three four-agent line-ups, with
100 rounds per line-up. Values below are mean official score per round, with the
reported standard error in parentheses.

| Match | First | Second | Third | Fourth |
| --- | --- | --- | --- | --- |
| Team agents with Official | Siegfried 6.66 (0.44) | SEAT-Q 3.90 (0.31) | Official 3.25 (0.28) | DQFD_RE 1.87 (0.15) |
| Team agents with RUEHL | Siegfried 5.46 (0.36) | SEAT-Q 3.98 (0.29) | RUEHL 3.72 (0.31) | DQFD_RE 1.95 (0.19) |
| Siegfried and SEAT-Q with both external agents | Siegfried 4.68 (0.32) | RUEHL 3.51 (0.26) | SEAT-Q 3.38 (0.22) | Official 2.23 (0.25) |

Siegfried had the highest observed mean score, survival rate, and kill count in
each of these line-ups. The comparisons support the team's tournament choice
within the recorded settings. They do not establish universal superiority,
component-level causal effects, or statistical significance between agents.

## Repository status

The three team models are stored in separate directories. Their READMEs
describe the available files, dependencies, and training or evaluation entry
points.

| Path | Contents | Documentation |
| --- | --- | --- |
| [`src/model_1/`](src/model_1/) | Siegfried inference code, trained checkpoint, dependencies, and verifier | [Siegfried README](src/model_1/README.md) |
| [`src/model_2/`](src/model_2/) | DQFD_RE development agents, training and pre-training code, demonstrations, and training records | [DQFD_RE README](src/model_2/README.md) |
| [`src/model_3/`](src/model_3/) | SEAT-Q variants, training and evaluation code, trained checkpoints, experiment records, and the game framework | [SEAT-Q README](src/model_3/README.md) |
| [`src/opponent agents/`](src/opponent%20agents/) | External Bomberman agents and reference repositories | Documentation within each external project |

The directory numbering identifies the team packages: Model 1 is Siegfried,
Model 2 is DQFD_RE, and Model 3 is SEAT-Q. SEAT-Q's internal `m2_*` version
names are explained in its README. External opponents are stored separately
from the team models.

## Published model: Siegfried

Siegfried is a stage-conditioned hybrid agent. Its learned component is a
Dueling Double DQN with 608,878 parameters in 19 tensors. The network processes
a 10-channel, 17 x 17 board tensor and 74 auxiliary values: 32 base features
plus 7 tactical features for each of six actions.

The deployed policy combines the Q-values with legal-action masking,
finite-horizon survival checks, tactical proposals, loop handling, and stage
routing.
The submitted weights come from the V18 checkpoint at 25,000 Task 3b
transitions. V19, V20, and final integration changed the controller without
retraining those weights.

The package provides `setup` and `act` for inference. It does not contain
training callbacks, optimizer state, target-network state, replay data, or
training logs. See [`src/model_1/README.md`](src/model_1/README.md) for its file
layout and task-specific behavior.

## Install and verify Siegfried

The pinned dependencies are NumPy 2.5.3 and the CPU build of PyTorch 2.11.0.
On Linux:

```bash
git clone https://github.com/Yimin-Yan/MLE_FinalProject.git
cd MLE_FinalProject
python3 -m venv .model1-venv
source .model1-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --requirement src/model_1/requirements.txt
python src/model_1/verify_delivery.py
```

On Windows PowerShell:

```powershell
git clone https://github.com/Yimin-Yan/MLE_FinalProject.git
Set-Location MLE_FinalProject
py -m venv .model1-venv
.\.model1-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --requirement .\src\model_1\requirements.txt
python .\src\model_1\verify_delivery.py
```

A valid package prints `"status": "PASS"`. The verifier checks the required
files, checkpoint digest, dependency versions, checkpoint structure, finite
parameters, and a CPU forward pass with output shape `[1, 6]`. This verifies
package integrity and inference compatibility; it is not a game-performance
test.

## Run Siegfried

Install the [official Bomberman RL framework](https://github.com/ukoethe/bomberman_rl),
then copy the complete `src/model_1` directory into the framework as
`agent_code/Siegfried`. Keep `model.pt` beside the Python modules.

From the framework directory, run a Task 4 game on CPU:

```bash
M2_DEVICE=cpu M2_TRAINING_STAGE=task4 \
python main.py play \
  --agents Siegfried rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 10 --no-gui
```

Windows PowerShell:

```powershell
$env:M2_DEVICE = "cpu"
$env:M2_TRAINING_STAGE = "task4"
python .\main.py play --agents Siegfried rule_based_agent rule_based_agent rule_based_agent --scenario classic --n-rounds 10 --no-gui
```

Supported stages are `task1`, `task2`, `task3a`, `task3b`, and `task4`. The
agent uses `task4` when `M2_TRAINING_STAGE` is unset. `M2_MODEL_PATH` can point
to another compatible checkpoint. Do not start this inference package with
`--train 1`.

## Run SEAT-Q

The SEAT-Q package includes the game framework, its reported variants, and
their training and evaluation scripts. Use Python 3.12 and run the following
commands from the repository root to evaluate the retained League policy:

```bash
python3 -m venv .seat-q-venv
source .seat-q-venv/bin/activate
python -m pip install --requirement src/model_3/requirements.txt
cd src/model_3
SEAT_D3QN_VARIANT=m2_4_c3_v2 python main.py play \
  --no-gui --agents seat_d3qn peaceful_agent coin_collector_agent rule_based_agent \
  --scenario classic --n-rounds 10 --seed 10001
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .seat-q-venv
.\.seat-q-venv\Scripts\Activate.ps1
python -m pip install --requirement .\src\model_3\requirements.txt
Set-Location .\src\model_3
$env:SEAT_D3QN_VARIANT = "m2_4_c3_v2"
python .\main.py play --no-gui --agents seat_d3qn peaceful_agent coin_collector_agent rule_based_agent --scenario classic --n-rounds 10 --seed 10001
```

All SEAT-Q commands run from `src/model_3`, which contains its `main.py`.
The [SEAT-Q README](src/model_3/README.md) maps the report's nine variants to
their source directories and lists training, checkpoint selection, and full
evaluation commands, together with the required source checkpoints and
instructions for preparing a fresh training copy.

## Evidence boundary

The shared matches contain one trained instance of each team model and three
selected line-ups. Their intervals describe evaluation uncertainty, not
variation across independently trained models. The report does not document a
shared seed schedule or starting-position balance for those matches. The
branch-specific tests use different protocols and remain separate from the
shared comparison.

Results therefore describe the tested agents and opponent compositions. They
do not isolate the contribution of a single feature, controller rule, training
stage, or architecture choice.

## Contributors

- Yimin Yan led the Siegfried branch and wrote the report.
- Junfeng Wei led the SEAT-Q branch and integrated the final report.
- Zhikai Zhang led the DQFD_RE branch and wrote the report.

All three team members contributed to model design and report writing.

## External code

The game environment and supplied baselines come from the
[Heidelberg Bomberman RL framework](https://github.com/ukoethe/bomberman_rl).
Material under `src/opponent agents/` comes from external sources. Review the
README and license included with each source before reuse.
