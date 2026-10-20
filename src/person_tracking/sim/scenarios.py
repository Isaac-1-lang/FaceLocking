"""Scripted real-world situations: head turns, occlusion, crossing, leaving, strangers."""
from .world import Occluder, Scenario, SimPerson

RED, BLUE, GREEN, YELLOW = (40, 40, 200), (200, 90, 30), (60, 170, 60), (40, 200, 220)
DARK, GREY, KHAKI = (60, 50, 40), (110, 110, 110), (110, 150, 170)
WALL = Occluder((540, 0, 860, 720))


def isaac(waypoints, yaw_keys=None, **kwargs):
    return SimPerson(1, 'Isaac', True, RED, DARK, waypoints, yaw_keys or [(0.0, 0.0)], **kwargs)


def alice(waypoints, yaw_keys=None, **kwargs):
    return SimPerson(2, 'Alice', True, BLUE, GREY, waypoints, yaw_keys or [(0.0, 0.0)], **kwargs)


def visitor(pid, waypoints, shirt=GREEN, yaw_keys=None, **kwargs):
    return SimPerson(pid, f'Visitor{pid}', False, shirt, KHAKI, waypoints, yaw_keys or [(0.0, 0.0)], **kwargs)


def head_turn():
    return Scenario(
        'head_turn', 'Target walks and turns away from the camera for ~5 s (face lost, body visible).',
        14.0, [isaac([(0, 380, 360, 560), (14, 800, 360, 560)],
                     [(0, 0), (3, 0), (3.8, 90), (4.5, 180), (7.0, 180), (7.7, 90), (8.5, 0)]),
               alice([(0, 180, 345, 540), (14, 180, 345, 540)])])


def long_turn():
    return Scenario(
        'long_turn', 'Target faces away for 12 s: confidence must decay to UNKNOWN, then recover on a new face match.',
        22.0, [isaac([(0, 450, 360, 560), (22, 650, 360, 560)],
                     [(0, 0), (3, 0), (4, 180), (15, 180), (16, 0)]),
               alice([(0, 1050, 345, 540), (22, 1050, 345, 540)])])


def occlusion():
    return Scenario(
        'occlusion', 'Target walks behind a wall and is fully hidden for ~1.6 s.', 16.0,
        [isaac([(0, 250, 360, 560), (2, 250, 360, 560), (14, 1050, 360, 560)]),
         alice([(0, 150, 345, 540), (16, 150, 345, 540)])], [WALL])


def long_occlusion():
    return Scenario(
        'long_occlusion', 'Target stands behind a wall for ~9 s: TEMPORARY must decay to UNKNOWN, no ghost lock.',
        20.0, [isaac([(0, 250, 360, 560), (3, 700, 360, 560), (13, 700, 360, 560), (16, 1050, 360, 560)]),
               alice([(0, 150, 345, 540), (20, 150, 345, 540)])], [WALL])


def path_crossing():
    return Scenario(
        'path_crossing', 'A closer stranger-in-blue crosses the target, hiding body and face for ~2 s.', 16.0,
        [isaac([(0, 250, 360, 560), (1, 250, 360, 560), (15, 1030, 360, 560)]),
         alice([(0, 1100, 390, 620), (1, 1030, 390, 620), (15, 250, 390, 620)])])


def crossing_same_colour():
    return Scenario(
        'crossing_same_colour', 'Crossing person wears the same red shirt: colour cannot disambiguate.', 16.0,
        [isaac([(0, 250, 360, 560), (1, 250, 360, 560), (15, 1030, 360, 560)]),
         SimPerson(3, 'Visitor3', False, RED, KHAKI,
                   [(0, 1100, 390, 620), (1, 1030, 390, 620), (15, 250, 390, 620)])])


def leave_return():
    return Scenario(
        'leave_return', 'Target leaves the frame for ~8 s and comes back: no ghost lock while away.', 24.0,
        [isaac([(0, 600, 360, 560), (2, 700, 360, 560), (6, 1500, 360, 560), (14, 1500, 360, 560),
                (18, 750, 360, 560)]),
         alice([(0, 200, 345, 540), (24, 200, 345, 540)])])


def strangers_only():
    return Scenario(
        'strangers_only', 'Target never appears; a lookalike occasionally fools the recogniser.', 15.0,
        [alice([(0, 250, 345, 540), (15, 900, 345, 540)]),
         visitor(3, [(0, 1000, 370, 580), (15, 300, 370, 580)], lookalike_of='Isaac')],
        lookalike_rate=0.04)


def crowd():
    return Scenario(
        'crowd', 'Four people wander, cross, and the target turns away twice.', 30.0,
        [isaac([(0, 300, 360, 560), (8, 700, 350, 560), (16, 1000, 360, 560), (24, 500, 365, 570),
                (30, 300, 360, 560)],
               [(0, 0), (5, 0), (6, 150), (9, 150), (10, 0), (18, 0), (19, 120), (22, 120), (23, 0)]),
         alice([(0, 1000, 380, 600), (10, 600, 385, 610), (20, 250, 380, 600), (30, 700, 380, 600)]),
         visitor(3, [(0, 150, 345, 540), (15, 600, 350, 550), (30, 1100, 345, 540)]),
         visitor(4, [(0, 1150, 350, 550), (12, 900, 340, 530), (30, 400, 340, 530)], shirt=YELLOW)],
        [Occluder((1180, 0, 1280, 720))])


SCENARIOS = {fn.__name__: fn for fn in (
    head_turn, long_turn, occlusion, long_occlusion, path_crossing,
    crossing_same_colour, leave_return, strangers_only, crowd)}


def get(name):
    try:
        return SCENARIOS[name]()
    except KeyError:
        raise ValueError(f'Unknown scenario {name!r}; choose from {", ".join(SCENARIOS)}') from None
