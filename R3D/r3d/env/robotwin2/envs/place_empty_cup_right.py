"""place_empty_cup, right arm only, coaster PINNED (sim baseline, CLAUDE.md "Sim baseline").

Two changes from `place_empty_cup`, everything else (grasp, placement, success
check against the coaster) inherited unchanged:

1. Spawn side fixed: the stock task draws `tag` in {0, 1} and `play_once` picks
   the arm from the cup's x sign; tag = 0 (cup x in [0.15, 0.3]) makes every
   episode a right-arm episode.
2. The coaster sits at ONE pose, the centre of its stock right-side range
   (x [-0.05, 0.1], y [-0.2, 0.05]). The target is then implicit, so the z floor
   of the crop box (sim_head_p1024.yml) can remove the 7 mm coaster entirely and
   no token is spent on a static target. With a moving coaster that crop would
   make the place location unobservable.
"""
from .place_empty_cup import place_empty_cup
from .utils import *


COASTER_XY = (0.025, -0.075)


class place_empty_cup_right(place_empty_cup):

    def load_actors(self):
        tag = 0  # stock: np.random.randint(0, 2); 0 = right side
        cup_xlim = [[0.15, 0.3], [-0.3, -0.15]]
        self.cup = rand_create_actor(
            self,
            xlim=cup_xlim[tag],
            ylim=[-0.2, 0.05],
            modelname="021_cup",
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
            convex=True,
            model_id=0,
        )
        cup_pose = self.cup.get_pose().p

        coaster_pose = rand_pose(
            xlim=[COASTER_XY[0], COASTER_XY[0]],
            ylim=[COASTER_XY[1], COASTER_XY[1]],
            rotate_rand=False,
            qpos=[0.5, 0.5, 0.5, 0.5],
        )
        # no overlap check needed: cup x >= 0.15 is >= 12.5 cm from the coaster
        self.coaster = create_actor(
            self,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            model_id=0,
            is_static=True
        )

        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area(self.coaster, padding=0.05)
        self.delay(2)
        cup_pose = self.cup.get_pose().p

    def point_label_roles(self):
        return {"object": [self.cup], "target": [self.coaster]}
