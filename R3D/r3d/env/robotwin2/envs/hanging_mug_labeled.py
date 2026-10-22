"""hanging_mug, UNCHANGED, plus the hooks the token-allocation analysis reads
(sim task screening, CLAUDE.md "Sim baseline"). Spawns, both-arm hand-over,
hanging and check_success are inherited from the stock task.

- point_label_roles(): mug = OBJECT, rack = TARGET (point_labels.py)
- critical_points(): the parts the success test is about -- the mug's functional
  point 0 (the handle it hangs by; check_success compares it with the hook) and
  the rack's functional point 0 (the hook). check_head_view.py counts the
  points / tokens / patch size within a radius of each.
"""
from .hanging_mug import hanging_mug


class hanging_mug_labeled(hanging_mug):

    def point_label_roles(self):
        return {"object": [self.mug], "target": [self.rack]}

    def critical_points(self):
        return {
            "mug_handle": list(map(float, self.mug.get_functional_point(0)[:3])),
            "rack_hook": list(map(float, self.rack.get_functional_point(0)[:3])),
        }
