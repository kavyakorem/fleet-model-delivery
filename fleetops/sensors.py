"""Deterministic simulated motor telemetry; faults are labeled at injection."""
import numpy as np

CHANNELS = ("current", "vibration", "temperature", "speed", "load")
STEPS = 24
WIDTH = 20

def sample(robot: int, sequence: int, fault: int | None = None):
    rng = np.random.default_rng(np.random.SeedSequence([robot, sequence, 2027]))
    load = rng.uniform(0.1, 0.95)
    unit = np.random.default_rng(robot + 500).normal(0, 1, 5)
    kind = rng.choice([0, 1, 2, 3], p=[.55, .15, .15, .15]) if fault is None else fault
    t = np.linspace(0, 1, STEPS)
    noise = rng.normal(0, 1, (STEPS, 5))
    x = np.column_stack([
        2.6 + 5.0*load + .26*unit[0] + .35*noise[:, 0],
        .30 + .32*load + .05*unit[1] + .07*noise[:, 1],
        37 + 15*load + 2.0*unit[2] + 1.3*noise[:, 2],
        1180 - 110*load + 15*unit[3] + 18*noise[:, 3],
        load + .02*noise[:, 4],
    ])
    severity = rng.uniform(.55, 1.45)
    if kind == 1:  # mechanical drag
        x[:, 0] += severity*(.8 + 1.1*t)
        x[:, 3] -= severity*(28 + 49*t)
    elif kind == 2:  # abnormal vibration
        x[:, 1] += severity*(.10 + .28*np.abs(np.sin(22*t)))
    elif kind == 3:  # overheating
        x[:, 2] += severity*(3 + 11*t)
    return x.astype(np.float32), int(kind != 0)

def features(windows):
    x = np.asarray(windows, dtype=np.float32)
    return np.concatenate([x.mean(axis=-2), x.std(axis=-2),
        x.max(axis=-2), x[..., -4:, :].mean(axis=-2)-x[..., :4, :].mean(axis=-2)], axis=-1)

def dataset(robots, windows=65):
    pairs = [sample(robot, index) for robot in robots for index in range(windows)]
    return features(np.stack([x for x, _ in pairs])), np.asarray([y for _, y in pairs])
