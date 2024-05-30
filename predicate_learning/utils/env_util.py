from Box2D import *
import numpy as np


############## Enviornment ################


def get_bounding_box(body):
    """
    Return the min max bounding box coordinate of a Box2D body
    """
    vertices = []
    for fixture in body.fixtures:
        if isinstance(fixture.shape, b2PolygonShape):
            vertices.extend([body.transform * v for v in fixture.shape.vertices])
        elif isinstance(fixture.shape, b2CircleShape):
            center = body.transform * fixture.shape.pos
            vertices.extend(
                [
                    center + [fixture.shape.radius] * 2,
                    center - [fixture.shape.radius] * 2,
                ]
            )
    vertices = np.array(vertices)

    return vertices.min(axis=0), vertices.max(axis=0)


def get_obj_size(body):
    """
    Return the 2D size of a Box2D body
    """
    p_min, p_max = get_bounding_box(body)
    return p_max - p_min


def get_obj_pos(body):
    """
    Get obj position as array.
    """
    return np.array(body.position)


# def sample_on_position(new_obj_size, receptacle, all_objects):
#     """
#     Sample collision-free position to place obj_1 on obj_2
#     Args:
#         new_obj_size: size of obj_1
#         receptacle: Box2D bodies
#         all_objects: all other objects in environment as a list of Box2D bodies
#     """
#     return sample_on_position_help_func(
#         new_obj_size,
#         receptacle.userData,
#         all_objects_boxes_dict={obj.userData: get_bounding_box(obj) for obj in all_objects},
#     )


def sample_on_position_help_func(
    new_obj_size, receptacle_name, all_objects_boxes_dict, num_sample=5, clearance=2.0, eps=0.01
):
    """
    Sample collision-free position to place new object on receptacle
    Args:
        new_obj_size: 2D size of new object
        receptacle: name of receptacle
        all_objects_boxes_dict: Dict[str, tuple(left_bottom, right_top)]
        num_samples: integer, number of sumples to draw
    """
    recep_min, recep_max = all_objects_boxes_dict[receptacle_name]

    # y coordinate aligns with top of obj_2
    y = recep_max[1]

    # sample collision-free x
    y_min = y
    y_max = y + new_obj_size[1]
    x_min = recep_min[0]
    x_max = recep_max[0]
    invalid_ranges = []
    for obj_name, (p_min, p_max) in all_objects_boxes_dict.items():
        if obj_name == receptacle_name:
            continue
        if p_min[1] > y_max or p_max[1] < y_min:
            continue
        if p_min[0] > x_max or p_max[0] < x_min:
            continue
        invalid_ranges.append((p_min[0], p_max[0]))

    invalid_ranges = sorted(invalid_ranges, key=lambda x: x[0])
    valid_ranges = []
    if len(invalid_ranges) == 0:
        valid_ranges.append((x_min + new_obj_size[0] / 2 - eps, x_max - new_obj_size[0] / 2 + eps))
    else:
        for i, x in enumerate(invalid_ranges):
            # first obstacle, sample on left side
            if i == 0:
                if x[0] - x_min > new_obj_size[0] + clearance:
                    valid_ranges.append(
                        (
                            x_min + new_obj_size[0] / 2 - eps,
                            x[0] - new_obj_size[0] / 2 - clearance,
                        )
                    )
            # sample between last and current obstacles
            elif x[0] - invalid_ranges[i - 1][1] > new_obj_size[0] + clearance * 2:
                valid_ranges.append(
                    (
                        invalid_ranges[i - 1][1] + new_obj_size[0] / 2 + clearance,
                        x[0] - new_obj_size[0] / 2 - clearance,
                    )
                )
            # last obstacle, sample on right
            if i == len(invalid_ranges) - 1:
                if x_max - x[1] > new_obj_size[0] + clearance:
                    valid_ranges.append(
                        (
                            x[1] + new_obj_size[0] / 2 + clearance,
                            x_max - new_obj_size[0] / 2 + eps,
                        )
                    )

    if len(valid_ranges) == 0:
        return None

    range_lens = [np.absolute(x[1] - x[0]) for x in valid_ranges]
    sample_prob = [x / sum(range_lens) for x in range_lens]
    samples = np.random.choice(len(sample_prob), num_sample, p=sample_prob).tolist()

    low = [valid_ranges[x][0] for x in samples]
    high = [valid_ranges[x][1] for x in samples]
    x_samples = np.random.uniform(low=low, high=high)

    positions = [(x, y) for x in x_samples]
    return positions


def check_on(obj_1, obj_2, eps=0.2):
    """
    Check if obj_1 is physically on obj_2
    Args:
        obj_1, obj_2: Box2D bodies
    """
    obj_2_min, obj_2_max = get_bounding_box(obj_2)
    obj_1_min, obj_1_max = get_bounding_box(obj_1)

    if np.absolute(obj_1_min[1] - obj_2_max[1]) < eps:
        if obj_1_min[0] > obj_2_max[0] or obj_1_max[0] < obj_2_min[0]:
            return False
        else:
            return True
    else:
        return False
