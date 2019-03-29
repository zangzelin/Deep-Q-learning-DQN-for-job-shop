import numpy as np
import random
import time
import matplotlib.pyplot as plt


class JobShop:
    # This class is the environment of Job shop problem

    bool_generate_random_jssp = None
    number_job = None
    number_machine = None
    number_features = None

    # the lower limit of one position of job 's processing time.
    time_low = None
    # the upper limit of one position of job 's processing time.
    time_high = None

    # Matrix of processing time, M_processing_time[i,j] is the processing time of job i 's position j.
    M_processing_time = None
    # Matrix of processing time, M_processing_order[i,j] is the machine restrain of job i 's position j.
    M_processing_order = None
    M_start_time = None
    M_end_time = None
    X_schedule_plan = None
    schedule_line = None

    def __init__(self, number_machine, number_job, time_low, time_high, bool_random):
        self.number_job = number_job
        self.bool_generate_random_jssp = random
        self.number_machine = number_machine
        self.time_low = time_low
        self.time_high = time_high
        self.schedule_line = []
        self.GenerateRandomProblem()

    def Get_Possible_Job_Position(self):
        # ergodic the schedule_line, and return the possible position to produce of jobs

        job_position_list = [0 for i in range(self.number_job)]
        for job_id, job_position in self.schedule_line:
            if job_position < self.number_machine-1:
                job_position_list[job_id] = job_position+1
            else:
                job_position_list[job_id] = -1

        return [[i, job_position_list[i]] for i in range(len(job_position_list))]

    def Get_Features(self, possible_job_position):
        # return the features of current state

        featrues = []
        for job_id, job_position in possible_job_position:
            f_item = self.GetFeature(job_id, job_position)
            featrues.append(f_item)

        return featrues

    def Step(self, action=None):
        # be called in main function
        # input action and return state score and done
        # action: choose a job to process.
        # state:

        done = False
        if action == None:
            self.MeasurementAction(self.schedule_line)
            possible_pob_position = self.Get_Possible_Job_Position()
            state = np.array(self.Get_Features(possible_pob_position))
            score = 0
        else:

            job_position_list = [0 for i in range(self.number_job)]
            for job_id, job_position in self.schedule_line:
                if job_position < self.number_machine-1:
                    job_position_list[job_id] = job_position+1
                else:
                    job_position_list[job_id] = -1
            if job_position_list[action] == -1:
                done = True
                canchoose = [[i, job_position_list[i]] for i in range(
                    self.number_job) if job_position_list[i] != -1]
                action = canchoose[0]
            else:
                action = [action, job_position_list[action]]

            self.schedule_line.append(action)
            self.MeasurementAction(self.schedule_line)
            # self.PlotResult()
            score = np.max(self.M_end_time)

            possible_pob_position = self.Get_Possible_Job_Position()
            state = np.array(self.Get_Features(possible_pob_position))

        state = [np.reshape(state[i], (1, 2,)) for i in range(self.number_job)]

        return state, score, done

    def GenerateRandomProblem(self):
        # Generate the jobshop problem
        # random problem or a stable problem

        if self.bool_generate_random_jssp == True:
            a = list(range(self.time_low, self.time_high))
            p = []
            for k in range(self.number_job):
                p.append(random.sample(a, self.number_machine))
            self.M_processing_time = np.array(p)

            a = list(range(self.number_machine))
            r = []
            for k in range(self.number_job):
                r.append(random.sample(a, self.number_machine))
            self.M_processing_order = np.array(r)

            sum_time_of_job = np.sum(self.M_processing_time, axis=1)

            for i in range(self.number_job):
                for j in range(i+1, self.number_job):
                    if sum_time_of_job[i] > sum_time_of_job[j]:
                        a = np.copy(self.M_processing_time[j, :])
                        self.M_processing_time[j,
                                               :] = self.M_processing_time[i, :]
                        self.M_processing_time[i, :] = a
                        sum_time_of_job[i], sum_time_of_job[j] = sum_time_of_job[j], sum_time_of_job[i]

            sum_time_of_mach = [[i, 0] for i in range(self.number_machine)]
            for i in range(self.number_job):
                for j in range(self.number_machine):
                    sum_time_of_mach[self.M_processing_order[i, j]
                                     ][1] += self.M_processing_time[i, j]

            for i in range(self.number_machine):
                for j in range(i+1, self.number_machine):
                    if sum_time_of_mach[i][1] > sum_time_of_mach[j][1]:
                        sum_time_of_mach[i], sum_time_of_mach[j] = sum_time_of_mach[j], sum_time_of_mach[i]

            nr = np.zeros((self.number_job, self.number_machine), dtype=int)-1
            for i in range(self.number_machine):
                nr[self.M_processing_order == i] = sum_time_of_mach[i][0]

            sum_time_of_mach = [[i, 0] for i in range(self.number_machine)]
            for i in range(self.number_job):
                for j in range(self.number_machine):
                    sum_time_of_mach[self.M_processing_order[i, j]
                                     ][1] += self.M_processing_time[i, j]

            self.M_processing_order = nr
        else:
            self.M_processing_order = np.array(
                [[1, 3, 0, 2], [0, 2, 1, 3], [3, 1, 2, 0], [1, 3, 0, 2], [0, 1, 2, 3]])
            self.M_processing_time = np.array([[18, 20, 21, 17], [18, 26, 15, 16], [
                17, 18, 27, 23], [18, 21, 25, 15], [22, 29, 28, 21]])

    def MeasurementAction(self, action_history):
        # measurement the action and return the makespan

        M_start_time = np.zeros((self.number_machine, self.number_job))
        M_end_time = np.zeros((self.number_machine, self.number_job))

        timeline_machine = np.zeros((self.number_machine), dtype=int)
        index_machine = np.zeros((self.number_machine), dtype=int)
        timeline_job = np.zeros((self.number_job), dtype=int)
        index_job = np.zeros((self.number_job), dtype=int)
        X_schedule_plan = np.zeros(
            (self.number_machine, self.number_job, 2), dtype=int)

        for job_id, job_position in action_history:
            machine_id = self.M_processing_order[job_id, job_position]

            current_start_time = max(
                timeline_machine[machine_id], timeline_job[job_id])
            current_end_time = current_start_time + \
                self.M_processing_time[job_id, job_position]

            timeline_machine[machine_id], timeline_job[job_id] = current_end_time, current_end_time
            current_index = index_machine[machine_id]
            M_start_time[machine_id, current_index] = current_start_time
            M_end_time[machine_id, current_index] = current_end_time
            X_schedule_plan[machine_id, current_index, :] = [
                job_id, job_position]
            index_machine[machine_id] += 1
            index_job[job_id] += 1

        self.M_start_time = M_start_time
        self.M_end_time = M_end_time
        self.X_schedule_plan = X_schedule_plan
        return np.max(M_end_time)

    def PlotResult(self, num=0):
        # plot function for the gant map

        colorbox = ['yellow', 'whitesmoke', 'lightyellow',
                    'khaki', 'silver', 'pink', 'lightgreen', 'orange', 'grey', 'r', 'brown']

        for i in range(100):
            colorArr = ['1', '2', '3', '4', '5', '6', '7',
                        '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
            color = ""
            for i in range(6):
                color += colorArr[random.randint(0, 14)]
            colorbox.append("#"+color)

        fig = plt.figure(figsize=(7, 4))
        for i in range(self.number_machine):
            # number_of_mashine:
            for j in range(self.number_job):
                # number_of_job:
                # % read the start time point
                mPoint1 = self.M_start_time[i, j]
                mPoint2 = self.M_end_time[i, j]  # % read the end time point
                mText = i + 1.5  # % read the index of machine
                PlotRec(mPoint1, mPoint2, mText)  # % plot subfunction
                Word = str(self.X_schedule_plan[i, j, 0]+1) + '.' + str(
                    self.X_schedule_plan[i, j, 1]+1)  # % read machine id

                x1, x2, x3, x4 = mPoint1, mPoint2, mPoint2, mPoint1
                y1, y2, y3, y4 = mText-0.8, mText-0.8, mText, mText
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4],
                         color=colorbox[self.X_schedule_plan[i, j, 0]])

                plt.text(0.5*mPoint1+0.5*mPoint2-3.5, mText-0.5, Word)

        plt.xlabel('Time')
        plt.ylabel('Machine')
        plt.tight_layout()
        plt.savefig('gant.png')
        plt.close()

    def Print_info(self):
        # print the problem infomation

        print('order')
        print(self.M_processing_order)
        print('time')
        print(self.M_processing_time)
        print('start time')
        print(self.M_start_time)
        print('end time')
        print(self.M_end_time)
        print('X')
        print(self.X_schedule_plan)

    def GetFeature(self, job_id, job_position):
        # get the feature of one position of one job 
        # readers can change the feature to get a more powerful model 

        # raw features
        machine_id = self.M_processing_order[job_id, job_position]
        job_time_need = np.sum(self.M_processing_time, axis=1)
        current_time_use = self.M_processing_time[job_id, job_position]

        machine_endtime = np.max(self.M_end_time, axis=1)
        job_endtime = np.sum(self.M_processing_time[job_id, :job_position])
        job_alltime = np.sum(self.M_processing_time[job_id, :])

        if job_position == 0:
            frac_currentend_othermachineave = 0.5
            frac_currentend_otherjobave = 0.5
            frac_currentendplusthisposition_othermachineave = 1
            schedule_finish_station = 0

            frac_jobposition_jobtime = 1
            frac_jobposition_totaltime = 1
        else:
            frac_currentend_othermachineave = (
                0.1+machine_endtime[machine_id]) / (0.1+np.average(machine_endtime))
            frac_currentendplusthisposition_othermachineave = (
                machine_endtime[machine_id]+current_time_use)/np.average(machine_endtime)
            schedule_finish_station = np.count_nonzero(
                self.M_end_time)/self.number_machine/self.number_job

            frac_currentend_otherjobave = (0.1+job_endtime) / (0.1+job_alltime)
            frac_jobposition_jobtime = current_time_use/job_time_need[job_id]
            frac_jobposition_totaltime = current_time_use/np.sum(job_time_need)

        # feature choose
        features = []
        # current features
        features.append(frac_currentend_othermachineave)
        features.append(frac_currentend_otherjobave)

        # features.append(frac_currentendplusthisposition_othermachineave)
        # features.append(schedule_finish_station)
        # # stable features
        # features.append(frac_jobposition_jobtime)
        # features.append(frac_jobposition_totaltime)

        self.number_features = len(features)

        if job_position == -1:
            features = [-1] * self.number_features

        return features


def PlotRec(mPoint1, mPoint2, mText):
    # sub function to plot a box in figure

    vPoint = np.zeros((4, 2))
    vPoint[0, :] = [mPoint1, mText-0.8]
    vPoint[1, :] = [mPoint2, mText-0.8]
    vPoint[2, :] = [mPoint1, mText]
    vPoint[3, :] = [mPoint2, mText]
    plt.plot([vPoint[0, 0], vPoint[1, 0]], [vPoint[0, 1], vPoint[1, 1]], 'k')
    plt.plot([vPoint[0, 0], vPoint[2, 0]], [vPoint[0, 1], vPoint[2, 1]], 'k')
    plt.plot([vPoint[1, 0], vPoint[3, 0]], [vPoint[1, 1], vPoint[3, 1]], 'k')
    plt.plot([vPoint[2, 0], vPoint[3, 0]], [vPoint[2, 1], vPoint[3, 1]], 'k')


if __name__ == "__main__":
    # main function used in debug

    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)

    problem = JobShop(4, 5, 15, 30, bool_random = False)
    # print(problem.MeasurementAction([]))
    print(problem.MeasurementAction([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                                     [0, 1], [1, 1], [2, 1], [3, 1], [4, 1], ]))
    problem.GetFeature(0, 0)
