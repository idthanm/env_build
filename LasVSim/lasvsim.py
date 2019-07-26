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


def sim_step(steps=None):
    if steps is None:
        steps = 1
    return simulation.sim_step(steps)


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


def get_ego_info():
    return simulation.get_ego_info()


def get_ego_road_related_info():
    return simulation.get_ego_road_related_info()


def set_ego(x, y, v, heading):
    simulation.agent.update_dynamic_state(x, y, v, heading)


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


def reset_simulation(overwrite_settings=None, init_traffic_path=None):
    """Reset simulation to it's initial state."""
    simulation.reset(simulation.settings, overwrite_settings, init_traffic_path)


def return_current_simulation_step():
    """Return current simulation step"""
    return simulation.tick_count













