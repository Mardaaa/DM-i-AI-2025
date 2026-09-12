"""Visual expert-driver demo. Use benchmark.py for fast headless evaluation."""

from expert import ExpertController
from settings import driver_config

_controller = ExpertController(driver_config())


def return_action(state):
    return _controller.actions(state)


if __name__ == "__main__":
    import pygame
    from benchmark import run_episode

    pygame.init()
    try:
        print(run_episode("565318", config=driver_config(), visual=True))
    finally:
        pygame.quit()