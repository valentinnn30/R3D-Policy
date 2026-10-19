"""stack_blocks_two, block spawn limited to the head camera's field of view
(sim task 2, CLAUDE.md "Sim baseline").

The ONLY change from the stock task: block x in [-0.24, 0.24] instead of
[-0.28, 0.28]. The stock aloha head camera sits at x = -0.032; projected through
its D435 intrinsics, its view at the spawn depth (cube centre z 0.766) spans
x ~[-0.34, +0.27], so the near-right spawn corner (x 0.28, y -0.08) left the
image. At 0.24 every cube, including its 2.5 cm edge, is in view. Both arms,
the |x| >= 0.05 rejection, the fixed first target (0, -0.13), play_once and
check_success are inherited unchanged.
"""
from copy import deepcopy

from .stack_blocks_two import stack_blocks_two
from .utils import *

BLOCK_X_MAX = 0.24


class stack_blocks_two_fov(stack_blocks_two):

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []
        for i in range(2):
            block_pose = rand_pose(
                xlim=[-BLOCK_X_MAX, BLOCK_X_MAX],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(block_pose):
                for j in range(len(block_pose_lst)):
                    if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-BLOCK_X_MAX, BLOCK_X_MAX],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def point_label_roles(self):
        # both cubes are the task objects; the first target is an implicit spot
        return {"object": [self.block1, self.block2]}
