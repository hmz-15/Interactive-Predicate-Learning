from Box2D import *
from Box2D.b2 import *
from predicate_learning.predicate_gym.Kitchen2D import Kitchen2D
from predicate_learning.predicate_gym.kitchen_constants import *
import numpy as np
import pygame

from predicate_learning.utils.env_util import get_bounding_box


class Kitchen2DEnv(Kitchen2D):
    def __init__(self, render, overclock=10):
        """
        A wrapper that helps create a customized Kitchen2D environment.
        Args:
            overclock: number of frames to skip when showing graphics. If overclock is None,
            this feature is not used
        """
        super().__init__(
            render,
            SINK_W,
            SINK_H,
            TABLE_THICK,
            SINK_POS_X,
            LEFT_TABLE_WIDTH,
            RIGHT_TABLE_WIDTH,
            FAUCET_H,
            FAUCET_W,
            FAUCET_D,
            planning=False,
            save_fig=False,
            liquid_name=LIQUID_NAME,
            liquid_frequency=LIQUID_FREQUENCY,
            overclock=overclock,
        )

        self.render = render

        # record table info
        self._table_info = {
            # "sink_bbox_xyxy": np.array(
            #     [SINK_POS_X, TABLE_HEIGHT - SINK_H, SINK_POS_X + SINK_W, TABLE_HEIGHT]
            # ),
            "table_range": np.array([-LEFT_TABLE_WIDTH + SINK_POS_X, SINK_POS_X]),
            "table_height": 0.0,
        }

        # to be filled
        self.objects = {}
        self.object_categories = {}

        self.body_to_color = {}

    def add_object(self, name, pos):
        body, color, category = self.create_object(self.world, name, pos)

        self.objects[name] = body
        self.object_categories[name] = category
        self.body_to_color[body] = color
        return body

    @staticmethod
    def create_object(world, name, pos):
        if name in BLOCK_LIKE_OBJS:
            body, color, category = create_block_like_object(world, name, pos)
        elif name in CONTAINER_OBJS:
            body, color, category = create_container_like_object(world, name, pos)
        elif name in STATIC_POLYGON_OBJS:
            body, color, category = create_static_polygon_object(world, name, pos)
        elif name in DYNAMIC_POLYGON_OBJS:
            body, color, category = create_dynamic_polygon_object(world, name, pos)
        else:
            raise NotImplementedError(f"Object {name} not supported!")

        return body, color, category

    @property
    def obj_states(self):
        """
        Return the states of objects (bboxes).
        """
        if self.render:
            surface = self.gui_world.screen
            surface_string = pygame.image.tostring(surface, "RGB")
            image = np.frombuffer(surface_string, np.uint8).reshape(
                (surface.get_height(), surface.get_width(), 3)
            )

        states = {}
        for name, body in self.objects.items():
            category = self.object_categories[name]
            bbox = get_bounding_box(body)
            obj_position = np.array([(bbox[0][0] + bbox[1][0]) / 2, (bbox[0][1] + bbox[1][1]) / 2])
            obj_size = np.array([bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]])

            states[name] = {"category": category, "position": obj_position, "size": obj_size}

            if self.render:
                # convert bbox into image coordinate (X, Y, W, H)
                # Box2D gui_world coordinate is at (SCREEN_WIDTH/2., TABLE_HEIGHT), y axis pointing up
                img_bbox = np.array(
                    [
                        (SCREEN_WIDTH / 2 + bbox[0][0]) * PPM,
                        (SCREEN_HEIGHT - (TABLE_HEIGHT + bbox[1][1])) * PPM,
                        (bbox[1][0] - bbox[0][0]) * PPM,
                        (bbox[1][1] - bbox[0][1]) * PPM,
                    ]
                )
                # crop image
                x, y, w, h = img_bbox.astype(np.int32)
                cropped_img = image[y : y + h, x : x + w, :].copy()

                states[name]["cropped_img"] = cropped_img

            # get particles if cup
            if body.userData == "cup" or "pot":
                num_particles = count_in_cup_particles(body, self.liquid.particles)
                states[name]["liquid_amount"] = num_particles / 10

        states = dict(sorted(states.items()))
        return states

    @property
    def table_info(self):
        return self._table_info


def create_block_like_object(world, name, pos):
    obj_info = BLOCK_LIKE_OBJS[name]
    w, h, color, userData = (
        obj_info["w"],
        obj_info["h"],
        obj_info["color"],
        obj_info["category"],
    )

    shift = np.array([w / 2.0, 0])
    body = world.CreateDynamicBody(position=pos, angle=0)
    polygon_shape = [(0, 0), (w, 0), (w, h), (0, h)]
    polygon_shape = [(v[0] - shift[0], v[1] - shift[1]) for v in polygon_shape]
    body.CreateFixture(shape=b2PolygonShape(vertices=polygon_shape), friction=1, density=0.05)
    body.usr_w = w
    body.usr_h = h
    body.usr_d = None
    body.shift = shift
    body.userData = userData

    return body, color, userData


def create_container_like_object(world, name, pos):
    obj_info = CONTAINER_OBJS[name]
    w, h, d, shifth, color, userData = (
        obj_info["w"],
        obj_info["h"],
        obj_info["d"],
        obj_info["shifth"],
        obj_info["color"],
        obj_info["category"],
    )

    shift = np.array([w / 2.0, shifth * h])
    body = world.CreateDynamicBody(position=pos, angle=0)
    polygon_shape = [(d / 2, 0), (w - d / 2, 0), (w - d / 2, d), (d / 2, d)]
    polygon_shape = [(v[0] - shift[0], v[1] - shift[1]) for v in polygon_shape]
    body.CreateFixture(shape=b2PolygonShape(vertices=polygon_shape), friction=1, density=0.5)
    polygon_shape = [(0, 0), (d, 0), (d, h), (0, h)]
    polygon_shape = [(v[0] - shift[0], v[1] - shift[1]) for v in polygon_shape]
    body.CreateFixture(shape=b2PolygonShape(vertices=polygon_shape), friction=1, density=0.5)
    polygon_shape = [(w, 0), (w, h), (w - d, h), (w - d, 0)]
    polygon_shape = [(v[0] - shift[0], v[1] - shift[1]) for v in polygon_shape]
    body.CreateFixture(shape=b2PolygonShape(vertices=polygon_shape), friction=1, density=0.5)
    body.usr_w = w * 1.0
    body.usr_h = h * 1.0
    body.usr_d = d * 1.0
    body.shift = shift
    body.userData = userData

    return body, color, userData


def create_static_polygon_object(world, name, pos):
    obj_info = STATIC_POLYGON_OBJS[name]
    polygon_shape, color, userData = (
        obj_info["polygon_shape"],
        obj_info["color"],
        obj_info["category"],
    )

    body = world.CreateStaticBody(
        position=pos,
        shapes=[b2PolygonShape(vertices=polygon_shape)],
        userData=userData,
    )
    return body, color, userData


def create_dynamic_polygon_object(world, name, pos):
    obj_info = DYNAMIC_POLYGON_OBJS[name]
    polygon_shape, usr_w, usr_h, color, userData = (
        obj_info["polygon_shape"],
        obj_info["usr_w"],
        obj_info["usr_h"],
        obj_info["color"],
        obj_info["category"],
    )

    body = world.CreateDynamicBody(position=pos, angle=0)
    body.CreateFixture(shape=b2PolygonShape(vertices=polygon_shape), friction=0.1, density=0.2)
    body.usr_w = usr_w
    body.usr_h = usr_h
    body.usr_d = None
    body.shift = None
    body.userData = userData

    return body, color, userData


def count_in_cup_particles(cup, particles):
    """
    Returns particles that are inside the cup.
    """
    incupparticles = []
    for p in particles:
        ppos = p.position - cup.position
        tp = cup.angle
        trans = np.array([[np.cos(tp), np.sin(tp)], [-np.sin(tp), np.cos(tp)]])
        ppos = np.dot(trans, ppos) + cup.shift

        if ppos[0] <= cup.usr_w and ppos[0] >= 0.0 and ppos[1] >= 0.0 and ppos[1] <= cup.usr_h:
            incupparticles += [p]

    return len(incupparticles)


if __name__ == "__main__":
    from predicate_learning.predicate_gym.Kitchen2D.kitchen2d.gripper import Gripper

    # get water and pour
    # env = Kitchen2DEnv(render=True, overclock=50)
    # cup = env.add_object("cup", (0, 0))
    # pot = env.add_object("pot", (-20, 0))
    # for _ in range(100):
    #     env.step()

    # gripper = Gripper(env, (0, 20), 0)

    # success = gripper.grasp(cup, 0.5)
    # if success:
    #     gripper.find_path(
    #         (gripper.position[0], gripper.position[1] + 4),
    #         0,
    #         maxspeed=1.0,
    #     )

    # gripper.get_liquid_from_faucet(6)

    # gripper.pour(pot, (3, get_obj_size(obj_to_pour)[1] / 2 + 10), 1.8)
    # env.destroy()

    # push plate on placemat
    env = Kitchen2DEnv(render=True, overclock=50)
    env.add_object("placemat", (0, 0))
    plate = env.add_object("plate", (-15, 0))
    for _ in range(100):
        env.step()

    gripper = Gripper(env, (0, 20), 0)
    gripper.push(plate, (1, 0), 0, 0.5)
