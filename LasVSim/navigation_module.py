from math import sqrt
import networkx as nx
import re
import copy

MAP_MIN_X=-933.0
MAP_MAX_X=933.0
MAP_MIN_Y=-933.0
MAP_MAX_Y=933.0
MAP_REGION_LEN=622.0
MAP_CROSS_WIDTH=36.0
MAP_ROAD_WIDTH=7.5
MAP_IN_FIELD=0
MAP_IN_ROAD=1
MAP_IN_CROSS=2

MISSION_GOTO_TARGET=0
MISSION_GOTO_CROSS=1
MISSION_TURNTO_ROAD=2

MISSION_START = -1
MISSION_RUNNING = 0
MISSION_COMPLETE = 1
MISSION_FAILED = 2

MISSION_LEFT_LANE = 'L'
MISSION_RIGHT_LANE = 'R'


class Map:
    """
    Map Modal of Autonomous Car Simulation System
    """
    def __init__(self):
        pass

    def get_cross_center(self,cross):
        """
        get crossing center position
        """
        cx,cy=cross
        return (float(cx)*MAP_REGION_LEN,float(cy)*MAP_REGION_LEN)

    def get_target(self,source,direction):
        """
        get target crosing info for source crossing and specified direction
        """
        cx,cy=source
        if direction == 'N':
            cy=cy+1
        elif direction == 'S':
            cy=cy-1
        elif direction == 'E':
            cx=cx+1
        elif direction == 'W':
            cx=cx-1
        else:
            return None
        if cx<-1 or cx>1 or cy<-1 or cy>1:
            return None
        else:
            return (cx,cy)

    def get_source(self,target,direction):
        """
        get source crosing info for target crossing and specified direction
        """
        cx,cy=target
        if direction == 'N':
            cy=cy-1
        elif direction == 'S':
            cy=cy+1
        elif direction == 'E':
            cx=cx-1
        elif direction == 'W':
            cx=cx+1
        else:
            return None
        if cx<-1 or cx>1 or cy<-1 or cy>1:
            return None
        else:
            return (cx,cy)

    def get_direction(self,source,target):
        """
        get direction from source crossing to target crossing
        """
        x1,y1=source
        x2,y2=target
        if x1<x2:
            return 'E'
        if x1>x2:
            return 'W'
        if y1<y2:
            return 'N'
        if y1>y2:
            return 'S'
        return None

    def map_position(self,x,y):
        """Get position status for point (x,y).

        Returns:
            vehicle's status: int variable indicating vehicle inside
                intersection, vehicle on road and vehicle out of map.
            center_cross: if vehicle inside intersection, return intersection's
                 center coordinates.
            lane_status: if vehicle on road, return current lane's source
                intersection's center coordinates, target intersection's center
                coordinates, heading direction and lane index as a dict.
        """
        if x<MAP_MIN_X or x>MAP_MAX_X or y<MAP_MIN_Y or y>MAP_MAX_Y:
            return (MAP_IN_FIELD,None)
        region_x,region_y=0,0
        if x>MAP_REGION_LEN/2:
            region_x=1
        elif x<-MAP_REGION_LEN/2:
            region_x=-1
        if y>MAP_REGION_LEN/2:
            region_y=1
        elif y<-MAP_REGION_LEN/2:
            region_y=-1
        x=x-region_x*MAP_REGION_LEN
        y=y-region_y*MAP_REGION_LEN
        center_cross=(region_x,region_y)
        if x>=-MAP_CROSS_WIDTH/2 and x<=MAP_CROSS_WIDTH/2 and y>=-MAP_CROSS_WIDTH/2 and y<=MAP_CROSS_WIDTH/2:
            return (MAP_IN_CROSS,center_cross)
        if x>=-MAP_ROAD_WIDTH and x<0:
            direction='S'
            if y<0:
                source=center_cross
                target=self.get_target(center_cross,direction)
            else:
                target=center_cross
                source=self.get_source(center_cross,direction)
            if x>=-MAP_ROAD_WIDTH/2:
                lane='L'
            else:
                lane='R'
            return (MAP_IN_ROAD,dict(source=source,target=target,direction=direction,lane=lane))
        elif x>=0 and x<=MAP_ROAD_WIDTH:
            direction='N'
            if y>0:
                source=center_cross
                target=self.get_target(center_cross,direction)
            else:
                target=center_cross
                source=self.get_source(center_cross,direction)
            if x<=MAP_ROAD_WIDTH/2:
                lane='L'
            else:
                lane='R'
            return (MAP_IN_ROAD,dict(source=source,target=target,direction=direction,lane=lane))
        elif y>=-MAP_ROAD_WIDTH and y<0:
            direction='E'
            if x>0:
                source=center_cross
                target=self.get_target(center_cross,direction)
            else:
                target=center_cross
                source=self.get_source(center_cross,direction)
            if y>=-MAP_ROAD_WIDTH/2:
                lane='L'
            else:
                lane='R'
            return (MAP_IN_ROAD,dict(source=source,target=target,direction=direction,lane=lane))
        elif y>=0 and y<=MAP_ROAD_WIDTH:
            direction='W'
            if x<0:
                source=center_cross
                target=self.get_target(center_cross,direction)
            else:
                target=center_cross
                source=self.get_source(center_cross,direction)
            if y<=MAP_ROAD_WIDTH/2:
                lane='L'
            else:
                lane='R'
            return (MAP_IN_ROAD,dict(source=source,target=target,direction=direction,lane=lane))
        return (MAP_IN_FIELD,None)

    def get_display_pos(self,pt,a):
        x,y=pt
        pos_status,pos_data=self.map_position(x,y)
        if pos_status == MAP_IN_ROAD:
            direction=pos_data['direction']
            return (x,y),-self.direction_to_angle(direction)
        else:
            return (x,y),a

    def get_road_lane_pos(self,road_lane):
        """
        get position info for a specified road lane
        """
        source=road_lane['source']
        target=road_lane['target']
        direction=road_lane['direction']
        lane=road_lane['lane']
        length=622.0-36
        base_on_source=1
        if source is not None:
            cx,cy=source
        elif target is not None:
            cx,cy=target
            base_on_source=0
        if direction in ['S','N']:
            h_pos=cx*622.0
            v_pos=cy*622.0
        else:
            h_pos=cy*622.0
            v_pos=cx*622.0
        if direction in ['N','W']:
            dis=3.75
        else:
            dis=-3.75
        if lane=='L':
            h_pos=h_pos+dis/2
        else:
            h_pos=h_pos+dis*3/2

        if direction in ['N','E']:
            if base_on_source==1:
                v_center=v_pos+622.0/2
            else:
                v_center=v_pos-622.0/2
        else:
            if base_on_source==1:
                v_center=v_pos-622.0/2
            else:
                v_center=v_pos+622.0/2

        if source is not None:
            v_source=v_center-length/2
        else:
            v_source=v_center
        if target is not None:
            v_target=v_center+length/2
        else:
            v_target=v_center

        if direction in ['S','W']:
            v_source,v_target=v_target,v_source

        return dict(center=h_pos,source=v_source,target=v_target)

    def is_same_road(self,r1,r2):
        return r1['source']==r2['source'] and r1['target']==r2['target'] and r1['direction']==r2['direction']

    def get_rough_direction(self,source,target):
        """
        get rough direction (N/E/W/S) from source point to target point
        """
        x1,y1=source[:2]
        x2,y2=target[:2]
        if abs(x1-x2)>abs(y1-y2):
            if x1<x2:
                return 'E'
            else:
                return 'W'
        else:
            if y1<y2:
                return 'N'
            else:
                return 'S'

    def direction_to_angle(self,direction):
        """
        get angle of direction (N/E/W/S)
        """
        if direction=='N':
            return 0.0
        elif direction=='W':
            return 90.0
        elif direction=='S':
            return 180.0
        else: # 'E'
            return 270.0

    def distance_to_cross_center(self,pos,cross):
        """
        get distance from position pos to the crossing center
        """
        x1,y1=pos[:2]
        x2,y2=[622.0*n for n in cross]
        return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

    def get_road_center_point(self,p):
        """
        get road center point for display
        """
        x,y=p
        if x<MAP_MIN_X or x>MAP_MAX_X or y<MAP_MIN_Y or y>MAP_MAX_Y:
            return (x,y)
        region_x,region_y=0,0
        if x>MAP_REGION_LEN/2:
            region_x=1
        elif x<-MAP_REGION_LEN/2:
            region_x=-1
        if y>MAP_REGION_LEN/2:
            region_y=1
        elif y<-MAP_REGION_LEN/2:
            region_y=-1
        dx=x-region_x*MAP_REGION_LEN
        dy=y-region_y*MAP_REGION_LEN
        if dx>=-MAP_CROSS_WIDTH/2 and dx<=MAP_CROSS_WIDTH/2:
            x=x-dx
        if dy>=-MAP_CROSS_WIDTH/2 and dy<=MAP_CROSS_WIDTH/2:
            y=y-dy
        return (x,y)

    def get_display_route(self, points, cross_list):
        """
        get polyline of the mission route for display
        """
        route = []
        route.append(self.get_road_center_point(points[0][:2]))
        for c in cross_list:
            route.append(self.get_cross_center(c))
        route.append(self.get_road_center_point(points[-1][:2]))
        return route

    def distance_in_direction(self,p1,p2,direction):
        """
        get projected distance in specified direction from p1 to p2
        """
        x1,y1=p1
        x2,y2=p2
        if direction in 'SN':
            return abs(y1-y2)
        else:
            return abs(x1-x2)

    def get_left_lane_center(self,p,direction):
        """
        get left road lane center point
        """
        x,y=self.get_road_center_point(p)
        if self.map_position(x,y)[0]==MAP_IN_CROSS:
            if direction is 'N':
                x += MAP_ROAD_WIDTH / 4
                y -= MAP_CROSS_WIDTH / 2
            elif direction is 'S':
                y += MAP_CROSS_WIDTH / 2
                x -= MAP_ROAD_WIDTH / 4
            elif direction is 'E':
                x -= MAP_CROSS_WIDTH / 2
                y -= MAP_ROAD_WIDTH / 4
            else:
                x += MAP_CROSS_WIDTH / 2
                y += MAP_ROAD_WIDTH / 4
        else:
            if direction is 'N':
                x+=MAP_ROAD_WIDTH/4
            elif direction is 'S':
                x-=MAP_ROAD_WIDTH/4
            elif direction is 'E':
                y-=MAP_ROAD_WIDTH/4
            else:
                y+=MAP_ROAD_WIDTH/4
        return x,y

    def get_nearest_lane(self, x, y, type):
        """
        get nearest road lane for mission edit
        """
        region_x,region_y=0,0
        if x>MAP_REGION_LEN/2:
            region_x=1
        elif x<-MAP_REGION_LEN/2:
            region_x=-1
        if y>MAP_REGION_LEN/2:
            region_y=1
        elif y<-MAP_REGION_LEN/2:
            region_y=-1
        dx=x-region_x*MAP_REGION_LEN
        dy=y-region_y*MAP_REGION_LEN

        if abs(dx)<abs(dy):
            if dx>0:
                dir='North'
                x=x-dx+MAP_ROAD_WIDTH*3/4
            else:
                dir='South'
                x=x-dx-MAP_ROAD_WIDTH*3/4
            if abs(dy)<(MAP_CROSS_WIDTH)/2+10:
                if y>0:
                    y=y-dy+(MAP_CROSS_WIDTH)/2+10
                else:
                    y=y-dy-((MAP_CROSS_WIDTH/2)+10)
        else:
            if dy>0:
                dir='West'
                y=y-dy+MAP_ROAD_WIDTH*3/4
            else:
                dir='East'
                y=y-dy-MAP_ROAD_WIDTH*3/4
            if abs(dx) < (MAP_CROSS_WIDTH) / 2 + 10:
                if x > 0:
                    x = x - dx + (MAP_CROSS_WIDTH) / 2 + 10
                else:
                    x = x - dx - ((MAP_CROSS_WIDTH / 2) + 10)

        status, info = self.map_position(x, y)
        if ((type is 'Target' and info['source'] is None) or
                (type is 'Source' and info['target'] is None)):
            dir0 = dir
            if dir0 is 'North':
                x-=MAP_ROAD_WIDTH*3/2
                dir='South'
            elif dir0 is 'South':
                x+=MAP_ROAD_WIDTH*3/2
                dir='North'
            elif dir0 is 'East':
                y += MAP_ROAD_WIDTH * 3 / 2
                dir = 'West'
            elif dir0 is 'West':
                y-=MAP_ROAD_WIDTH*3/2
                dir='East'
        return x, y, dir

    def get_out_cross_center(self,source,direction):
        """
        get cross center in the outside of the map
        """
        cx,cy=source
        if direction is 'N':
            cy+=1
        elif direction is 'S':
            cy-=1
        elif direction is 'E':
            cx+=1
        elif direction is 'W':
            cx-=1
        return self.get_cross_center((cx,cy))


class Mission:
    """
    Mission Modal for Autonomous Car Simulation System
    """
    map=None
    source = None
    target = None
    cross_list = []
    pos = None
    pos_status = None
    status = MISSION_START
    current_task = None
    current_lane = None
    speed_limit = 60.0/3.6
    cross_task=None
    next_direction = None

    engine_state = (0, 0, 1)  # acc m/s^2, engSpeed r/s, gearRatio
    control_input = (0, 0, 0)  # engTorque N*m, brakePressure N, steerWheel deg


    def __init__(self, map, points, type=None):
        """
        mission initialization with map, source, and target
        """
        self.mission_type = type
        self.points = points
        self.pos = self.points[0]
        self.last_pos = None
        self.map = map
        self.status = MISSION_RUNNING
        self.pos_status, self.current_lane = self.map.map_position(
            self.points[0][0], self.points[0][1])
        self.states, self.lanes = [], []
        self.target_direction = []
        for i in range(len(points)):
            s, lane = self.map.map_position(self.points[i][0],
                                                 self.points[i][1])
            assert s == MAP_IN_ROAD, 'Selected location is not on any road.'
            self.states.append(s)
            self.lanes.append(lane)
            self.target_direction.append(lane['direction'])

        self.cross_list = []
        for i in range(len(self.states)-1):
            if (self.map.is_same_road(self.lanes[i+1], self.lanes[i]) and
                        self.map.get_rough_direction(self.points[i],
                                                     self.points[i+1]) ==
                        self.lanes[i]['direction']):
                continue
            else:
                cross1 = self.lanes[i]['target']
                cross2 = self.lanes[i + 1]['source']
                self.cross_list += self.get_cross_path(cross1, cross2,
                                                       self.lanes[i]['direction'],
                                                       self.lanes[i+1]['direction'])
        self.loop_cross_list = copy.deepcopy(self.cross_list)  # Save original cross list used for loop mission.

        if len(self.cross_list) > 0:
            if len(self.cross_list) > 1:
                next_direction = self.map.get_direction(self.cross_list[0],
                                                        self.cross_list[1])
            else:
                next_direction = self.map.map_position(self.points[-1][0],
                                                       self.points[-1][1])[1]['direction']
            self.next_direction = next_direction
            lane = self.get_lane(self.lanes[0]['direction'], next_direction)
            self.current_task = (MISSION_GOTO_CROSS, (self.cross_list[0], lane))
            self.speed_limit = 60.0/3.6
        else:
            self.current_task = (MISSION_GOTO_TARGET, self.points[-1])
            self.speed_limit = 60.0/3.6

    def get_cross_path(self,source,target,source_dir,target_dir):
        """Get mission route.

        Returns:
            A list consists of several intersection's index.
        """

        assert(source is not None and target is not None)
        G=nx.DiGraph()
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in range(4):
                    d='EWSN'[k]
                    _d='WENS'[k]
                    s=(i,j)
                    t=self.map.get_target(s,d)
                    if t is not None:
                        if s==source and _d==source_dir:
                            continue
                        if t==target and _d==target_dir:
                            continue
                        G.add_edge(s,t)
        path_list=[p for p in nx.all_shortest_paths(G,source,target)]
        if len(path_list)==0:
            return None
        elif len(path_list)==1:
            return path_list[0]

        best_path=None
        count_left=100
        for path in path_list:
            dirs=source_dir
            for i in range(len(path)-1):
                dirs=dirs+self.map.get_direction(path[i],path[i+1])
            dirs=dirs+target_dir
            count=len(re.findall('(?=(NW|WS|SE|EN))',dirs))
            if count<count_left:
                best_path=path
                count_left=count
        return best_path

    def get_lane(self,dir1,dir2):
        """Get road lane type for route plan.

        Returns:
            'L' or 'R'.
        """
        if (dir1=='N' and dir2=='E')\
                or (dir1=='E' and dir2=='S')\
                or (dir1=='S' and dir2=='W')\
                or (dir1=='W' and dir2=='N'):
            return MISSION_RIGHT_LANE
        else:
            return MISSION_LEFT_LANE

    def arrive_target(self):
        """
        check if the mission is complete
        """
        if self.mission_type == 'Loop':
            return False
        else:
            if self.last_pos is None:
                self.last_pos=self.pos
                return False
            if self.status==MISSION_COMPLETE:
                return True
            if self.current_task[0]!=MISSION_GOTO_TARGET:
                self.last_pos=self.pos
                return False
            x,y,v,a=self.pos
            xt,yt,vt,at=self.points[-1]
            if abs(x-xt)<2 and abs(y-yt)<2 and v==0:
                self.status = MISSION_COMPLETE
                return True
            if abs(x-xt)<2 and abs(y-yt)<2 and abs(v-vt)<1:
                self.status = MISSION_COMPLETE
                return True
            # if abs(x-xt)<1 and abs(y-yt)<1 and abs(v-vt)<1:
            #     return True
            x0,y0,v0,a0=self.last_pos
            if self.target_direction[-1] in 'NS':
                s1,s2,st=y0,y,yt
            else:
                s1,s2,st=x0,x,xt
            if self.target_direction[-1] in 'SW':
                s1,s2=s2,s1
            if s1<=st and st<=s2:
                self.status = MISSION_COMPLETE
                return True
            self.last_pos=self.pos
            return False

    def update(self, pos):
        """
        update mission status, get current task
        """
        if self.status == MISSION_RUNNING:
            #  Get current pos and lane info.
            self.pos = pos
            x, y, v, a = pos
            new_status, pos_data = self.map.map_position(x, y)
            if new_status == MAP_IN_FIELD:  # Vehicle ran out of road.
                self.status = MISSION_FAILED
                return
            if new_status == self.pos_status:  # Same as last pos status.
                if new_status == MAP_IN_ROAD:
                    self.current_lane = pos_data
                if self.current_task[0] == MISSION_GOTO_TARGET:
                    if self.arrive_target():
                        self.status = MISSION_COMPLETE
            else:  # Pos status changed.
                self.pos_status=new_status
                if self.pos_status==MAP_IN_ROAD:  # Vehicle enters road.
                    self.current_lane=pos_data
                    self.cross_list=self.cross_list[1:]
                    if len(self.cross_list) > 0:
                        current_direction=pos_data['direction']
                        if len(self.cross_list)>1:
                            next_direction=self.map.get_direction(
                                self.cross_list[0],
                                self.cross_list[1])
                        else:
                            next_direction = self.map.map_position(
                                self.points[-1][0],
                                self.points[-1][1])[1]['direction']
                        self.next_direction=next_direction
                        lane=self.get_lane(current_direction,next_direction)
                        self.current_task=(MISSION_GOTO_CROSS,(self.cross_list[0],lane))
                        self.speed_limit=60.0/3.6
                    else:
                        if self.mission_type == 'One Way':
                            self.current_task = (MISSION_GOTO_TARGET, self.points[-1])
                            self.speed_limit=60.0/3.6
                        elif self.mission_type == 'Loop':
                            self.cross_list = copy.deepcopy(self.loop_cross_list)
                            current_direction = pos_data['direction']
                            if len(self.cross_list) > 1:
                                next_direction = self.map.get_direction(
                                    self.cross_list[0], self.cross_list[1])
                            else:
                                next_direction = self.map.map_position(
                                    self.points[-1][0],
                                    self.points[-1][1])[1]['direction']
                            self.next_direction = next_direction
                            lane = self.get_lane(current_direction,
                                                 next_direction)
                            self.current_task = (
                            MISSION_GOTO_CROSS, (self.cross_list[0], lane))
                            self.speed_limit = 60.0 / 3.6

                elif self.pos_status==MAP_IN_CROSS:  # Vehicle enters cross.
                    if len(self.cross_list)>1:
                        next_cross=self.cross_list[1]
                        current_cross=pos_data
                        direction=self.map.get_direction(current_cross,next_cross)
                        if direction is None:
                            self.status=MISSION_FAILED
                            return
                        self.current_task=(MISSION_TURNTO_ROAD,dict(source=current_cross,target=next_cross,direction=direction))
                    else:
                        s,next_road=self.map.map_position(self.points[-1][0],self.points[-1][1])
                        if s!=MAP_IN_ROAD:
                            self.status=MISSION_FAILED
                            return
                        self.current_task=(MISSION_TURNTO_ROAD,next_road)
                        direction=next_road['direction']
                    if self.current_lane['direction']==direction:
                        self.speed_limit=60.0/3.6
                        self.cross_task='S'
                    else:
                        self.speed_limit=20.0/3.6
                        if self.current_lane['lane']==MISSION_LEFT_LANE:
                            self.cross_task='L'
                        else:
                            self.cross_task='R'

    def get_current_task(self):
        return self.current_task

    def get_status(self):
        return self.status

    def get_description(self):
        """
        get description of mission status for display
        """
        if self.status == MISSION_START:
            return 'start'
        elif self.status==MISSION_COMPLETE:
            return 'complete'
        elif self.status==MISSION_FAILED:
            return 'failed'
        elif self.status==MISSION_RUNNING:
            if self.current_task[0]==MISSION_GOTO_CROSS:
                return 'go forward to cross'
            elif self.current_task[0]==MISSION_GOTO_TARGET:
                return 'go forward to target'
            elif self.current_task[0]==MISSION_TURNTO_ROAD:
                if self.cross_task=='S':
                    return 'pass crossing'
                elif self.cross_task=='L':
                    return 'turn left'
                elif self.cross_task=='R':
                    return 'turn right'
                else:
                    return 'unavailable'
            else:
                return 'unavailable'
        else:
            return 'unavailable'
