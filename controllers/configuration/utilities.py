import math

import numpy as np


def normalize_to_range(value, min, max, newMin, newMax):
    value = float(value)
    min = float(min)
    max = float(max)
    newMin = float(newMin)
    newMax = float(newMax)
    return (newMax - newMin) / (max - min) * (value - max) + newMax


def get_distance_from_target(robot_node, target_node):
    robotCoordinates = robot_node.getField('translation').getSFVec3f()
    targetCoordinate = target_node.getField('translation').getSFVec3f()

    dx = robotCoordinates[0] - targetCoordinate[0]
    dy = robotCoordinates[1] - targetCoordinate[1]
    dz = robotCoordinates[2] - targetCoordinate[2]
    distanceFromTarget = []
    distanceFromTarget.append(math.sqrt(dx * dx +  dy * dy + dz * dz))
    distanceFromTarget.append(dx)
    distanceFromTarget.append(dy)
    distanceFromTarget.append(dz)
    distanceFromTarget.append(targetCoordinate[0])
    distanceFromTarget.append(targetCoordinate[1])
    distanceFromTarget.append(targetCoordinate[2])
    distanceFromTarget.append(robotCoordinates[0])
    distanceFromTarget.append(robotCoordinates[1])
    distanceFromTarget.append(robotCoordinates[2])
    return distanceFromTarget


def get_angle_from_target(robot_node,
                          target_node,
                          is_true_angle=False,
                          is_abs=True):

    robotAngle = robot_node.getField('rotation').getSFRotation()[3]

    robotCoordinates = robot_node.getField('translation').getSFVec3f()
    targetCoordinate = target_node.getField('translation').getSFVec3f()

    x_r = (targetCoordinate[0] - robotCoordinates[0])
    z_r = (targetCoordinate[2] - robotCoordinates[2])

    z_r = -z_r

    # robotWorldAngle = math.atan2(robotCoordinates[2], robotCoordinates[0])

    if robotAngle < 0.0: robotAngle += 2 * np.pi

    x_f = x_r * math.sin(robotAngle) - \
          z_r * math.cos(robotAngle)

    z_f = x_r * math.cos(robotAngle) + \
          z_r * math.sin(robotAngle)

    # print("x_f: {} , z_f: {}".format(x_f, z_f) )
    if is_true_angle:
        x_f = -x_f
    angleDif = math.atan2(z_f, x_f)

    if is_abs:
        angleDif = abs(angleDif)

    return angleDif
