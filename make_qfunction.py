import numpy as np

nodes = np.array([[0,2],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[2,4],[3,1],[3,2],[3,3],[4,2]])
obs_nodes = np.array([[1,2],[2,1],[2,3],[3,2]])
bit = 23

def get_state(state):
    q = np.array([2, 0, 0, 0, -1], dtype=int)
    state = format(state, '0' + str(bit) + 'b')
    agv = state[:13]
    obs = state[13:17]
    rot = int(state[17:19], 2)
    passing = state[19:]

    for action in range(5):
        # ゴールの向き
        if action!=0 and action==rot:
            q[action] = 1
        # 衝突
        if action==0 and (agv[2]=='1' or obs[0]=='1'):
            q[action] = -2
        if action==1 and (agv[5]=='1' or obs[1]=='1'):
            q[action] = -2
        if action==2 and (agv[10]=='1' or obs[3]=='1'): 
            q[action] = -2
        if action==3 and (agv[7]=='1' or obs[2]=='1'):
            q[action] = -2
        if action==4 and agv[6]==1:
            q[action] = -2
        # 前方干渉
        if action==0 and passing[0]=='1':
            q[action] = -2
        if action==1 and passing[1]=='1':
            q[action] = -2
        if action==2 and passing[2]=='1':
            q[action] = -2
        if action==3 and passing[3]=='1':
            q[action] = -2
    return q

if __name__ == '__main__':
    max = 2 ** bit
    q_table = np.zeros((max, 5), dtype=int)

    for state in range(max):
        if state % 100000 == 0:
            print('Episode', state)
        q_table[state] = get_state(state)
    
    np.save('yaml/qfunction', q_table)
    print(np.array([2., 0., 0., 0., -0.5], dtype=float))