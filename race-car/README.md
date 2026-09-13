# Race Car

## Deterministic expert solution

An autonomous, hand-written controller is now included: no neural network,
training data, learned weights, seed lookup, or simulator-state access.

- [expert.py](expert.py): geometric sensor reconstruction, traffic tracking,
  bang-bang steering and short-horizon planning.
- [example.py](example.py): watch the expert drive in Pygame.
- [benchmark.py](benchmark.py): fast headless evaluation using the original physics.
- [optimize.py](optimize.py): reproducible parameter sweeps.
- [optuna_search.py](optuna_search.py): Bayesian parameter search with crash
    penalties, independent selection/test seeds, and resumable SQLite storage.
- [EXPERT.md](EXPERT.md): mathematics, usage, results and limitations.
- [MANEUVER.md](MANEUVER.md): optional two-stage maneuver planner with recovery
    checks, adaptive batching and end-of-game scoring. On 50 new paired seeds it
    improved mean distance **13.4%** at the same observed 49/50 completion rate,
    but decisions were much slower. The existing Optuna preset remains deployed.
- [DRIFT.md](DRIFT.md): first 500k-oriented increment: exact distance-loss
    accounting and optional accelerate-while-drifting paths. Higher mean distance
    on development seeds came with **more crashes**; this variant is research-only,
    not deployed, and has not reached 500k.

**Optuna held-out results:** 98/100 complete games, mean distance **201,242**,
best **413,092**. On the same 100 fresh seeds the prior defaults averaged
148,559 and finished 95/100: **35.5% more mean distance**.
The API/demo now use [configs/expert.json](configs/expert.json).
See [OPTUNA.md](OPTUNA.md) and [the full report](results/optuna-v1/holdout.json).
These are local results, not a guarantee of safety or a competition score.
Under the current 3600-tick rules, the absolute distance ceiling is **684,180**;
one million is not physically attainable even with uninterrupted acceleration.

🔴 Ready

🟡 Set

🟢 GO! 

🏁 It's racing time! 🏁 

Race against the competition to go the furthest in the allotted time, but be careful, one small crash can end your run!

![Race Car](../images/race_car_intro.png)

## About the game
You control the yellow car. Red and blue cars will spawn in random lanes - it is your job to dodge them. The supplied implementation enables 16 evenly spaced sensors by default, each with a 1000px reach; sensors can be removed during initialization. Figure 2 shows an image of the sensors with names.

Each tick the game is updated. The game runs with 60 ticks per second. A list stores future actions, and each tick, an action is popped from the list and applied to the car. If there are no actions in the list, it will repeat the last action. If there is no last action, it will default to 'NOTHING'. 

A game runs for up to 60 seconds - 3600 ticks. If you crash, the game will immediately end. Once the game ends, the final score will be the distance you achieved. 

### Environment
The game runs with 5 lanes of equal size. Your car will spawn in the center lane, while other cars will spawn randomly in other lanes. The other cars will not leave their lane, and only one other car can be in each lane at a time. They can spawn in front or behind your car, and they spawn with a speed relative to yours. Their speed varies over time. 

On the top and bottom of the screens are walls. If you hit the walls your car will crash, so no off-roading in this one. 

### Your Goal

Your goal is to go as far as you can in one minute. Your game will **end** if you crash into other cars or into walls. Your final score will be based on your distance.

Interpret the sensor input and respond with commands for your car. The included expert controller does this without training.

### Controls

Pygame provides local visualization. The example now runs the expert driver; the original arrow-key controller remains in [src/game/core.py](src/game/core.py).

To communicate with the server for validation and evaluation, use the functions found in dtos.py. You can test if these work using the *test connection* button on [cases.dmiai.dk](https://cases.dmiai.dk). 

When the competition server needs actions, it will request them from your server. To reduce network delays, send a batch of actions (not just one) in each response.


### Sensors

Sensor output is your information from the game. By default there are 16 sensors on the car, each positioned at a specific angle (in degrees) relative to the center of the car with a reach of 1000 pixels. The image below shows the sensors, as well as a list of all sensors.

![Sensors](../images/race_car_sensors.png)

**List of Sensors (angle, name):**

| Angle   | Name               |
|---------|--------------------|
| 0       | left_side          |
| 22.5    | left_side_front    |
| 45      | left_front         |
| 67.5    | front_left_front   |
| 90      | front              |
| 112.5   | front_right_front  |
| 135     | right_front        |
| 157.5   | right_side_front   |
| 180     | right_side         |
| 202.5   | right_side_back    |
| 225     | right_back         |
| 247.5   | back_right_back    |
| 270     | back               |
| 292.5   | back_left_back     |
| 315     | left_back          |
| 337.5   | left_side_back     |

Each sensor is positioned at the specified angle (in degrees) relative to the center of the car and has a reach of 1000 pixels

## Scoring

Your score will be based on your distance. Scores will be normalised, lowest will recieve 0 and highest 1. Only scores above the baseline will count. If your score is below the baseline, it will auromatically get 0. 

## Validation and Evaluation
To test your model and server connection, start a validation attempt. You can only have one attempt going at once, but attempts are unlimited. Your attempt will be put into a queue, and run when it's your turn. The validation attempts will use random seeds. We recommend testing your network delay using the validation attempts and optimizing your model and server to fit. 

Once you are ready to evaluate your final model, start your evaluation attempt. You only have **ONE** try, so make sure the model is ready for the final test. Your score from the evaluation is the one you will be judged on. 

The evaluation opens up on Thursday the 7th at 12:00 CET and will have a preset seed.

## Quickstart

```cmd
git clone https://github.com/amboltio/DM-i-AI-2025
cd DM-i-AI-2025/race-car
python -m pip install -r requirements-dev.txt
```


### Serve your endpoint
Serve your endpoint locally and test that everything starts without errors

```cmd
python api.py
```
Open a browser and navigate to http://localhost:9052. You should see a message stating that the endpoint is running. 
Feel free to change the `HOST` and `PORT` settings in `api.py`. 

You can send the following action responses:
- NOTHING
- ACCELERATE
- DECELERATE
- STEER_RIGHT
- STEER_LEFT


### Run the simulation locally
```cmd
python example.py
```
By default the expert drives autonomously with the Optuna-selected settings.
For a fast headless run of that same preset, use
`python benchmark.py --config configs/expert.json --games 10`.
Omitting `--config` deliberately benchmarks the old baseline defaults.
For tests, use `python -m pytest tests -q`.


**We recommend you do not change the amount of lanes or the size of the game during training.**