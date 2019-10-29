from centralized_env.central_decision import CentralDecisionEnv
import time

height, width = 30, 30
vehNum = 8
vehModelList = [0, 2, 3, 4, 6, 8, 9, 10]


CrossRoad = CentralDecisionEnv(vehNum, vehModelList, height, width, 4)

for count in range(10000):
    collisionFlag = False
    endFlag = False
    tag = 0
    CrossRoad.reset()
    # CrossRoad.reStart()
    print()
    while not endFlag:
        action = [0] * vehNum
        [state, reward, endFlag, _] = CrossRoad.step(action)
        CrossRoad.render()
        tag += 1
        # print(count, "step: ", tag, "collision?: ", collisionFlag, "end?: ", endFlag)
        # print(reward)
        #print(state)
        #time.sleep(2)


