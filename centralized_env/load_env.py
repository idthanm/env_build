from baselines.common.cmd_util import make_vec_env
from baselines.common.vec_env import VecNormalize
import pickle
from matplotlib import pyplot as plt


def render(N, width, height, laneWidth, vehs):
    plt.cla()
    plt.title("Demo")
    ax = plt.axes(xlim=(-width - 3, width + 3), ylim=(-height - 3, height + 3))
    plt.axis("equal")
    argument = 1.5
    plt.plot([laneWidth, laneWidth, laneWidth + argument, width],
             [- height, - laneWidth - argument, - laneWidth, - laneWidth], "k")
    plt.plot([laneWidth, laneWidth, laneWidth + argument, width],
             [height, laneWidth + argument, laneWidth, laneWidth], "k")
    plt.plot([- laneWidth, - laneWidth, - laneWidth - argument, - width],
             [- height, - laneWidth - argument, - laneWidth, - laneWidth], "k")
    plt.plot([- laneWidth, - laneWidth, - laneWidth - argument, - width],
             [height, laneWidth + argument, laneWidth, laneWidth], "k")

    plt.plot([0.0, 0.0], [- height, height], ":k")
    plt.plot([- width, width], [0.0, 0.0], ":k")

    lines = []

    for i in range(N):
        if not vehs[i][0]:
            line = ax.plot([], [], "k")[0]
            lines.append(line)
            lines[-1].set_data(vehs[i][1], vehs[i][2])

    safeLine = []
    for i in range(N):
        if not vehs[i][0]:
            line = ax.plot([], [], ":r")[0]
            safeLine.append(line)
            safeLine[-1].set_data(vehs[i][3], vehs[i][4])

    # for i in range(N):
    #     if not vehs[i][0]:
    #         plt.plot(vehs[i][5], vehs[i][6], "go")

    plt.pause(100)

pkl_file = open('envlist.pkl', 'rb')
envlist = pickle.load(pkl_file)
pkl_file.close()
# for timestep in range(len(envlist)):
#     centralized_env = envlist[timestep]
#     N = centralized_env[0]
#     width = centralized_env[1]
#     height = centralized_env[2]
#     laneWidth = centralized_env[3]
#     vehs = []
#     for i in range(N):
#         vehs.append(centralized_env[4][i])
#     render(N, width, height, laneWidth, vehs)

env = envlist[20]
N = env[0]
width = env[1]
height = env[2]
laneWidth = env[3]
vehs = []
for i in range(N):
    vehs.append(env[4][i])
render(N, width, height, laneWidth, vehs)



