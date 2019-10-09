from LasVSim.simulator import *

simulation = None


def create_simulation(path=None):
    """Create a LasVSim simulation.

    Args:
        path: Simulation setting file path.
    """
    global simulation
    simulation = Simulation(path)
    return simulation


def load_scenario(path):
    """Load a simulation setting file.

    Args:
        path: Simulation setting file path.
    """
    simulation.load_scenario(path)

def seed(seed):  # this should be called before every time you reset lasvsim
    """
    :param seed:
    :return: set simulation seed
    """
    simulation.set_seed(seed)


def sim_step(steps=None):
    if steps is None:
        steps = 1
    return simulation.sim_step(steps)


# def sim_step_internal(steps=None):
#     """Run simulation for given steps.
#
#     This function only updates traffic and ego dynamic states. If no decision
#     output or controller output is given, then the last step value will be used.
#
#     Args:
#         steps: Simulation steps, int.
#     """
#     if steps is None:
#         steps = 1
#     return simulation.sim_step_internal(steps)


def get_all_objects():
    """Get all objects' info at current step.

    Returns:
         A list containing each object's info at current step. For example:

         [{'type': 0, 'x': 0.0, 'y': 0.0, 'v': 0.0, 'angle': 0.0, 'rotation': 0,
         'winker': 0, 'winker_time': 0, 'length': 0.0, 'width': 0.0,
         'height': 0.0, 'radius': 0.0}, ...]
         The order of objects is constant at every step..
    """
    return simulation.get_all_objects()


def get_detected_objects():
    """Get object's info which is detected by ego vehicle at current step.

    Returns:
         A list containing each detected object's info at current step.
         For example:

         [(id, x, y, v, a, w, h), (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),...]
         The id of each objects is constant at every step..
    """
    return simulation.get_detected_objects()


def get_ego_position():
    """Get ego vehicle's 2-d geo position.

    Returns:
        A list consists of x coordinate and y coordinate. For example:

        [23.442, 23.44] The first indicates x coordinate.
    """
    return simulation.get_ego_position()

def get_self_car_info():
    """Get ego vehicle's 2-d geo position.

    Returns:
        A list consists of x coordinate and y coordinate. For example:

        [23.442, 23.44] The first indicates x coordinate.
    """
    return simulation.get_self_car_info()


def set_engine_torque(torque):
    """Set ego vehicle's engine output torque in N*m.
    """
    simulation.agent.engine_torque = torque
    simulation.agent.controller.set_track(
        engine_torque=simulation.agent.engine_torque,
        brake_pressure=simulation.agent.brake_pressure,
        steering_wheel=simulation.agent.steer_wheel)


def set_brake_pressure(brake):
    """Set ego vehicle's brake system pressure in Mpa."""
    simulation.agent.brake_pressure = brake
    simulation.agent.controller.set_track(
        engine_torque=simulation.agent.engine_torque,
        brake_pressure=simulation.agent.brake_pressure,
        steering_wheel=simulation.agent.steer_wheel)


def set_steer_angle(steer):
    """Set ego vehicle's steering wheel angle in deg."""
    simulation.agent.steer_wheel = steer
    simulation.agent.controller.set_track(
        engine_torque=simulation.agent.engine_torque,
        brake_pressure=simulation.agent.brake_pressure,
        steering_wheel=simulation.agent.steer_wheel)


def set_delta_steer_and_acc(acc, delta_steer):
    """Set ego vehicle's delta steering wheel angle in deg and acceleration in m/s^2."""
    simulation.agent.steer_wheel += delta_steer
    simulation.agent.acceleration = acc
    simulation.agent.controller.set_track(
        steering_wheel=simulation.agent.steer_wheel,
        acceleration=simulation.agent.acceleration)


def set_ego_position(x, y, v, heading):
    simulation.agent.x = x
    simulation.agent.y = y
    simulation.agent.v = v
    simulation.agent.heading = heading


def save_simulation_data(path):
    """Save simulation data."""
    simulation.export_data(path)


def export_simulation_data(path):
    """Export simulation data in csv format.

    This csv file can be loaded by LasVSim-gui to replay the simulation."""
    if simulation is None:
        print('No simulation loaded.')
        return False
    simulation.export_data(path)


def get_bias():
    """Return ego vehicle's distance to the center of it;s current lane.

    Returns:
        double, m."""
    return simulation.traffic.get_dis2center_line()  # 左正右负


def reset_simulation(overwrite_settings=None, init_traffic_path=None):
    """Reset simulation to it's initial state."""
    simulation.reset(simulation.settings, overwrite_settings, init_traffic_path)


def return_current_simulation_step():
    """Return current simulation step"""
    return simulation.tick_count













