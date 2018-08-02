# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:00:38 2017

@author: Administrator
"""
import random
import time
import copy
import numpy
import multiprocessing
import Queue
import sys
# record the start time
starttime = time.time()

# variables for storing file's info
node_size = 0
Graph = {}
Task = []
event_name = ""
depot_node = 0
required_edge_size = 0
nonrequired_node_size = 0
car_size = 0
capacity = 0
total_cost_of_required_edge = 0
dijstra_dict = {}

# for multithreading
process_queue = Queue.Queue()
total_child_pop = []
total_child_fit = []
total_child_cost = []

def read_file(file_name):

    '''
    This function is for IO operation
    :param filename is the path
    '''

    file_object2 = open(file_name, 'r')
    try:
        lines = file_object2.readlines()
    finally:
        file_object2.close()
    global Graph, Task, node_size, event_name, depot_node, required_edge_size
    global nonrequired_node_size, car_size, capacity, total_cost_of_required_edge
    event_name = lines[0].split(":")[1].strip()
    node_size = int(lines[1].split(":")[1].strip())
    depot_node = int(lines[2].split(":")[1].strip())
    required_edge_size = int(lines[3].split(":")[1].strip())
    nonrequired_node_size = int(lines[4].split(":")[1].strip())
    car_size = int(lines[5].split(":")[1].strip())
    capacity = int(lines[6].split(":")[1].strip())
    total_cost_of_required_edge = int(lines[7].split(":")[1].strip())
    for line in lines[9:-1]:
        nums = line.split()
        Graph[(int(nums[0]), int(nums[1]))] = (int(nums[2]), int(nums[3]))
        Graph[(int(nums[1]), int(nums[0]))] = (int(nums[2]), int(nums[3]))
        if not int(nums[3]) == 0:
            Task.append((int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])))


def get_current_node(min_cost, notVisited):

    '''
    This function is a priavte method for dijkstra.
    :param min_cost is the min_cost dictionary
    :param notVisited is the set for not visited node
    return the not-visited node which has the min cost currently
    '''

    min_node = [0, float('inf')]
    for key in min_cost.keys():
        if key[1] in notVisited:
            if min_cost[key] < min_node[1]:
                min_node[0] = key[1]
                min_node[1] = min_cost[key]
    return min_node[0]


def get_current_neighbor(current_node):

    '''
    This function is a private method for dijkstra
    :param current_node is the current node
    return all its neighbors info as a dictionar type
    '''

    neighbor = {}
    for key in Graph.keys():
        if key[0] == current_node:
            neighbor[key] = Graph[key]
    return neighbor


def single_dijsktra(startNode):

    '''
    This function is to to generate a min cost path for a given start node
    :parem startNode means the given start node
    return a dictionary whose key is start and end node and whose value is min cost
    '''

    # Initialize visited list
    notVisited = set()
    for i in range(node_size):
        notVisited.add(i + 1);
    # Initalize min_cost list
    min_cost = {}
    for i in range(node_size):
        if not i + 1 == startNode:
            min_cost[(startNode, i + 1)] = float('inf')
        else:
            min_cost[(startNode, i + 1)] = 0

    # do dijsktra
    while not len(notVisited) == 0:
        current_node = get_current_node(min_cost, notVisited)
        notVisited.remove(current_node)
        neighbor_dict = get_current_neighbor(current_node)
        for key in neighbor_dict.keys():
            if key[1] in notVisited:
                if min_cost[(startNode, current_node)] + neighbor_dict[key][0] < min_cost[(startNode, key[1])]:
                    min_cost[(startNode, key[1])] = min_cost[(startNode, current_node)] + neighbor_dict[key][0]
    return min_cost


def get_total_mincost_dict():

    '''
    This function will generate a total min_cost dictionary for the whole graph
    return a dictionary of min cost
    '''

    global dijstra_dict
    for i in range(node_size):
        dijstra_dict.update(single_dijsktra(i + 1))


def split(chromosome,dep_node,cap,dijstra):

    '''
    This function will used for split a total tour into sections which has min cost
    This split method is the second phase of Ulusoy's algorithm
    :param dijstra_dict is a dictionary contains the shortest path\
    :param chromosome is a given ordered task list
    return cost,car_information
    '''

    # Vi means the cost on the shortest path from depot to node i
    global dijstra_dict
    global depot_node
    global capacity
    capacity = cap
    dijstra_dict = dijstra
    depot_node = dep_node

    TASK_START = 0
    TASK_END = 1
    TASK_COST = 2
    TASK_DEMAND = 3
    V_set = []
    size = len(chromosome)
    for i in range(size):
        V_set.append(float('inf'))
    V_set[0] = 0
    # Pre i means the pre node of task i on the shortest path from depot to node i
    Pre_number = []
    Car_number = []
    for i in range(size):
        Pre_number.append(0)
        Car_number.append(0)

    for counter in range(size - 1):
        i = counter + 1
        load = 0
        cost = 0
        j = i
        while (j < size and load <= capacity):
            load += chromosome[j][TASK_DEMAND]
            if i == j:
                cost = dijstra_dict[(depot_node, chromosome[j][TASK_START])] + chromosome[j][TASK_COST] + \
                       dijstra_dict[(chromosome[j][TASK_END], depot_node)]
            else:
                cost = cost - dijstra_dict[(chromosome[j - 1][TASK_END], depot_node)] + \
                       dijstra_dict[(chromosome[j - 1][TASK_END], chromosome[j][TASK_START])] +\
                       chromosome[j][TASK_COST] + dijstra_dict[(chromosome[j][TASK_END], depot_node)]

            if load <= capacity:
                Vnew = V_set[i - 1] + cost
                if Vnew < V_set[j]:
                    V_set[j] = Vnew
                    Pre_number[j] = i - 1
                j += 1

    a = len(Pre_number) - 1
    counter = 1
    while not a == 0:
        index = Pre_number[a] + 1
        while index <= a:
            Car_number[index] = counter
            index += 1
        counter += 1
        a = Pre_number[a]
    Car = [0]

    counter = 1
    i = 1
    while i < len(Car_number) - 1:
        if Car_number[i] == Car_number[i + 1]:
            Car.append(counter)
        else:
            Car.append(counter)
            counter += 1
        i += 1
    Car.append(counter)

    return V_set[size - 1], Car


def route_list(chromosome, car_set):
    '''
    This function will used for generating a routes list for split method
    :param chromosome is an ordered task list with route divison
    :param car_set is the infomation about how to divide route
    return ordered route list
    '''

    result = []
    index = [1]
    i = 1
    j = 1
    while i < len(car_set):
        while i <= j and j < len(car_set):
            j += 1
            if j < len(car_set) and car_set[i] != car_set[j]:
                i = j
                index.append(j)
        break
    index.append(None)

    k = 0
    while k < len(car_set):
        for j in range(len(index)):
            if k < j:
                result.append(chromosome[index[k]:index[j]])
                k = j
        break
    return result


def greedy_mincost(Task_segement):

    '''
    This function will used for ordering a task list in greedy way and satisfy the capacity
    This method will select next task according to the cost between current task to the rest unchecked task.
    If there exists same cost's task, random select one by applying a tiny random float
    :param Task_segement is an out-of-ordered task list
    return ordered route list
    '''

    temp = copy.deepcopy(Task_segement)
    a = []
    # bi-direction task
    for ele in temp:
        a.append((ele[1], ele[0], ele[2], ele[3]))
    temp += a
    inital = depot_node
    total_route = []
    route = []
    point_mark = inital
    del_task = 0
    current_capacity = 0

    while temp != []:
        min = float('inf')
        for index, task in enumerate(temp):
            cost = dijstra_dict[(inital, task[0])] + random.uniform(0, 0.0001)
            if cost < min:
                min = cost
                point_mark = task[1]
                del_task = task

        current_capacity += del_task[3]
        if current_capacity <= capacity:
            route.append(del_task)
            inital = point_mark
            temp.remove((del_task[1], del_task[0], del_task[2], del_task[3]))
            temp.remove(del_task)

        elif current_capacity > capacity:
            current_capacity = 0
            total_route.append(route)
            inital = depot_node
            route = []

    total_route.append(route)

    return total_route


def routes_assess(routes):

    '''
    This function will assess the total cost of input routes and violation of capacity
    :param routes is the total routes, stored in a list
    return current total cost and violated capacity
    '''

    violation = []
    cost = 0
    over_capacity = 0
    for i, route in enumerate(routes):
        curcost, real_capacity = single_route_assess(route)
        cost += curcost
        if real_capacity - capacity > 0:
            violation.append((i, real_capacity - capacity))
        else:
            violation.append((i, 0))

    for i, element in enumerate(violation):
        over_capacity += element[1]

    return cost, over_capacity


def single_route_assess(route):
    '''
    This function is a private method for routes assess, it can assess a single route
    :param route is a given route, stored in a list
    return current total cost and real capacity
    '''

    cost = 0
    real_capacity = 0
    for j, element in enumerate(route):
        if j == 0:
            cost += (dijstra_dict[(depot_node, element[0])] + element[2])
        else:
            cost += (dijstra_dict[(route[j - 1][1], element[0])] + element[2])
        real_capacity += element[3]
    cost += dijstra_dict[(route[-1][1], depot_node)]

    return cost, real_capacity


def first_population(original_size):
    '''
    This funcation is for generate first population\
    First, get many greedy chromosome and select top K as first population
    :param size is population size
    return result the population and the corresponding fitness and total cost including the cost of capacity violation
    '''
    # random generate chromosome
    population = []
    population_fitness = []
    population_total_cost = []
    total = []
    temp_task = Task
    timea = time.time()
    while time.time()-timea < 5:
        tem_task = greedy_mincost(temp_task)
        if tem_task not in population:
            fitness, overcapacity = routes_assess(tem_task)
            total.append((fitness, tem_task, fitness))
            total.sort()
    if len(total)>original_size:
        for item in total[:original_size]:
            population_fitness.append(item[0])
            population.append(item[1])
            population_total_cost.append(item[2])
    else:
        for item in total:
            population_fitness.append(item[0])
            population.append(item[1])
            population_total_cost.append(item[2])

    # print "first best",population_fitness[0],time.time()-starttime
    return population,population_fitness,population_total_cost


def select_parent(population, population_fitness):
    '''
    The is private method is for random selecting parents
    based on fitness function
    :param population is the chromosome set
    :param population_fitness is the corresponding fitness value
    return the select chromosome
    '''

    sumfit = max(population_fitness) - min(population_fitness) + 1
    maxval = max(population_fitness)
    fit_possibility = [float(maxval - x) / sumfit for x in population_fitness]
    x = random.uniform(0, sum(fit_possibility))
    cumulative_probability = 0.0
    for item, item_probability in zip(population, fit_possibility):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item, population_fitness[population.index(item)]


def big_cross_over(par1,par2):

    '''
    This function is big crossover
    It converts the routes list into a single route regardless the capacity limit
    and random select two position do OX and then use split to generate an ideal route list in this order
    :param p1 and p2 are parents
    return the offspring
    '''

    p1 = []
    p2 = []
    for route in par1:
        p1+=route
    for route2 in par2:
        p2+=route2
    size = len(p1)
    point1 = random.randrange(1, size)
    point2 = random.randrange(0, point1)
    while point2 == point1:
        point2 = random.randrange(0, point1)

    part = p1[0:point2] + p1[point1:]
    size = len(p1[0:point2])
    canzhao = p2[0:point2] + p2[point1:]
    index = []
    for i in range(len(part)):
        if part[i] in canzhao:
            canzhao.remove(part[i])
        else:
            index.append(i)
    counter = 0
    for i in range(len(index)):
        part[index[i]] = canzhao[counter]
        counter += 1
    child = part[:size] + p2[point2:point1] + part[size:]
    fitness,car = split([(0,0,0,0)]+child,depot_node,capacity,dijstra_dict)
    newchild = route_list([(0,0,0,0)]+child,car)

    return newchild,fitness,0


def small_cross_over(r1,r2):

    '''
    This function is small crossover
    Random select two routes and then do OX.
    After cross over may cause capacity violation
    :param r1 and r2 are parents
    return the offspring
    '''

    routes1 = copy.deepcopy(r1)
    routes2 = r2
    every_task = []
    for route in r1:
        for ele in route:
            every_task.append(ele)
            every_task.append((ele[1],ele[0],ele[2],ele[3]))

    route1_index = random.randrange(0, len(routes1))
    route2_index = random.randrange(0, len(routes2))

    sub1 = routes1[route1_index]
    sub2 = routes2[route2_index]
    sub1_index = random.randint(0, len(sub1) - 1)
    sub2_index = random.randint(0, len(sub2) - 1)
    routes1.remove(sub1)
    sub1 = sub1[:sub1_index]+sub2[sub2_index:]
    for i,route in enumerate(routes1):
        for j,ele in enumerate(route):
            if ele in every_task:
                every_task.remove(ele)
                every_task.remove((ele[1],ele[0],ele[2],ele[3]))
    dele = []
    for i,task in enumerate(sub1):
        if task in every_task:
            every_task.remove((task[1], task[0], task[2], task[3]))
            every_task.remove(task)
        else:
            dele.append(task)
    for i in dele:
        sub1.remove(i)

    routes1 = routes1[:route1_index]+[sub1]+routes1[route1_index:]

    while len(every_task)!=0:
        task = every_task.pop()
        route_index = random.randrange(0,len(routes1))
        route = routes1[route_index]
        insert_index = random.randrange(0,len(route))
        route.insert(insert_index,task)
        every_task.remove((task[1], task[0], task[2], task[3]))


    for ele in routes1:
        if len(ele) ==0:
            routes1.remove(ele)

    fitness,over_capacity = routes_assess(routes1)

    return routes1,fitness,over_capacity


def single_insertion(child):

    '''
    This function is for local search and applied in mutation part
    Random select one task and randomly insert it into certain route in certain position
    :param child is the result of crossover
    return the neighbor of child
    '''

    child1 = copy.deepcopy(child)

    car1_index = random.randrange(0, len(child1))
    r1 = child1[car1_index]
    while len(r1)<=1:
        car1_index = random.randrange(0, len(child1))
        r1 = child1[car1_index]

    car2_index = random.randrange(0, len(child1))
    r2 = child1[car2_index]

    task1_index = random.randrange(0, len(r1))
    task2_index = random.randrange(0, len(r2))

    new_gene1 = r1[task1_index]
    r2.insert(task2_index,new_gene1)
    r1.remove(new_gene1)
    if len(r1)==0:
        child1.remove(r1)
    currentcost, over_capacity = routes_assess(child1)

    return currentcost,child1,over_capacity

def swap(child):

    '''
    This function is for local search and applied in mutation part
    Random swap two task
    :param child is the result of crossover
    return the neighbor of child
    '''

    child1 = copy.deepcopy(child)

    car1_index = random.randrange(0, len(child1))
    car2_index = random.randrange(0, len(child1))
    r1 = child1[car1_index]
    r2 = child1[car2_index]
    task1_index = random.randrange(0, len(r1))
    task2_index = random.randrange(0, len(r2))
    new_gene1 = r1[task1_index]
    r1[task1_index] = r2[task2_index]
    r2[task2_index] = new_gene1
    currentcost,over_capacity = routes_assess(child1)

    return currentcost,child1,over_capacity


def double_insertion(child):

    '''
    This function is for local search and applied in mutation part
    Random select two consequent tasks and randomly insert it into certain route in certain position
    :param child is the result of crossover
    return the neighbor of child
    '''

    child1 = copy.deepcopy(child)

    car1_index = random.randrange(0, len(child1))
    r1 = child1[car1_index]
    while len(r1)<=2:
        car1_index = random.randrange(0, len(child1))
        r1 = child1[car1_index]

    car2_index = random.randrange(0, len(child1))
    r2 = child1[car2_index]

    task1_index = random.randrange(0, len(r1)-1)
    task2_index = random.randrange(0, len(r2))

    new_gene1 = r1[task1_index]
    new_gene2 = r1[task1_index+1]
    r2.insert(task2_index,new_gene1)
    r2.insert(task2_index+1,new_gene2)
    r1.remove(new_gene1)
    r1.remove(new_gene2)
    if len(r1)==0:
        child1.remove(r1)
    currentcost, over_capacity = routes_assess(child1)

    return currentcost,child1,over_capacity


def merge(child,penalty):

    '''
    This function is for local search and applied in mutation part
    It is called MS operator and can break the tie in local search
    Random select two routes and merge them up
    Sort them by repeatly greedy alogrithm and do split after finding correct order
    Then connect the new routes with the rest
    :param child is the result of crossover
    return the neighbor of child
    '''

    i = random.randrange(1, len(child))
    j = random.randrange(0, i)
    while i == j:
        j = random.randrange(0, i)

    route = copy.deepcopy(child)

    new_chromosome = []
    a1 = route[i]
    new_chromosome += a1
    b1 = route[j]
    new_chromosome += b1
    route.remove(a1)
    route.remove(b1)
    candidate = []
    for i in range(5):
        temp = greedy_mincost(new_chromosome)
        fitness, overcapacity = routes_assess(temp)
        candidate.append((fitness+overcapacity*penalty,temp,fitness,overcapacity))
    candidate.sort()
    new_chromosome = candidate[0][1]

    chromosome = []
    for item in new_chromosome:
        chromosome += item

    currentcost,car = split([(0, 0, 0, 0)] + chromosome,depot_node,capacity,dijstra_dict)
    child1 = route_list([(0, 0, 0, 0)] + chromosome,car)
    child1 += route
    currentcost, over_capacity = routes_assess(child1)


    return currentcost,child1,over_capacity


def mutate(child1,fitness,mutate_possib,penalty):

    '''
    This function is for mutate
    It applies four local seach method and save the best result
    :param child is the result of crossover
    :param fitness is the cost result of crossover
    :param mutate_possib is the possibility of mutation
    :param penalty is the penalty parameter for capacity violation
    return the best of child
    '''

    local_min = fitness
    local_cap = routes_assess(child1)[1]
    local_child = child1
    if random.uniform(0,1)<mutate_possib:
        results = []
        min_cost, local_child, cap = single_insertion(child1)
        results.append((min_cost + cap * penalty, min_cost, local_child, cap, "single"))
        min_cost, local_child, cap = swap(child1)
        results.append((min_cost+cap*penalty,min_cost,local_child,cap,"swap"))
        min_cost, local_child, cap = double_insertion(child1)
        results.append((min_cost+cap*penalty,min_cost,local_child,cap,"double"))
        min_cost, local_child, cap = merge(child1,penalty)
        results.append((min_cost, min_cost, local_child, cap, "merge"))
        results.sort()

        if results[0][0]<local_min:
            local_child = results[0][2]
            local_min = results[0][1]
            local_cap = results[0][3]

    return local_child,local_min,local_cap


def add_children(population,population_fitness,size,mutate_possib,cap,dijstra,dep_node,penalty,switch,seed):

    '''
    This function describes the process of crossover and mutate
    :param population is about family situation
    :param population_fitness is the corresponding fitenss
    :param size is the scale of children population
    :param mutate_possib is the possibility of mutation
    the rest paramter is for multiprocessing
    return children population
    '''

    global dijstra_dict
    global depot_node
    global capacity
    capacity = cap
    dijstra_dict = dijstra
    depot_node = dep_node
    random.seed(seed)
    children_fitness = []
    children_population = []
    children_totalcost = []
    counter = 1
    while len(children_fitness) < size:

        parent1, pafitns1 = select_parent(population, population_fitness)
        parent2, pafitns2 = select_parent(population, population_fitness)

        # determine use which cross over method
        if switch == True:
            child, fitness_old, over_capacity = small_cross_over(parent1, parent2)
        else:
            child, fitness_old, over_capacity = big_cross_over(parent1, parent2)

        child,fitness_old,over_capacity = mutate(child,fitness_old,mutate_possib,penalty)
        child.sort()

        # avoid clone
        if child not in children_population and child not in population:
            children_population.append(child)
            children_fitness.append(fitness_old)
            children_totalcost.append(fitness_old + over_capacity * penalty)
            counter+=1

    return_tuple = (children_population,children_fitness,children_totalcost)
    return return_tuple


def shrink(children_population,children_fitness,child_cost,parent_population,parent_fitness,parent_cost,size,percent):

    '''
    This function describes the process of filter the total popluation into given size.
    During the shrink, a threshold means the number of infeasible solutions we accept
    and rank all the solution based on the cost
    :param children_* are the assessment variables and population of children
    :param parent_* are he assessment variables and population of parents
    :param percent means how many infeasible solutions we accept
    the rest paramter is for multiprocessing
    return children population
    '''

    total_fitness = children_fitness+parent_fitness
    total_population = children_population + parent_population
    total_cost = child_cost+parent_cost
    prori =[]
    current_best_fitness = 0
    current_best_popluation = 0
    for i in range(len(total_fitness)):
        prori.append((total_cost[i], total_fitness[i], total_population[i]))
    prori.sort()
    for item in prori:
        if item[0] == item[1]:
            current_best_popluation = item[2]
            current_best_fitness = item[1]
            break
        else:
            continue

    if len(total_fitness)<=size:
        return total_population,total_fitness,total_cost,current_best_popluation,current_best_fitness
    else:
        new_population = []
        new_fitness = []
        new_cost = []
        valid_con = 0
        elastic_con = 0
        for i in prori:
            if i[0] == i[1] and i[2] not in new_population:
                valid_con += 1
            elif i[0] != i[1] and elastic_con<(size*percent) and i[2] not in new_population:
                elastic_con += 1
            else:
                continue
            new_population.append(i[2])
            new_fitness.append(i[1])
            new_cost.append(i[0])
            if valid_con+elastic_con==size:
                break
        return new_population,new_fitness,new_cost,current_best_popluation,current_best_fitness



def muti_process(process_num, queue):

    '''
    This function for multiprocessing operation
    :param process_num is process number
    :param queue contains the paramters for sub threading methods
    '''

    pool = multiprocessing.Pool(process_num)
    while not queue.empty():
        work = queue.get()
        pool.apply_async(func=add_children, args=work, callback=update_child)
    pool.close()
    pool.join()



def update_child(return_tuple):

    '''
    This function collects and updates the children list after each sub-processing
    :param return_tuple is the result of add children method
    '''

    global total_child_fit
    global total_child_pop
    global total_child_cost
    total_child_pop.extend(return_tuple[0])
    total_child_fit.extend(return_tuple[1])
    total_child_cost.extend(return_tuple[2])

def to_format(chromosome):

    '''
    This function can format the solution to standard model
    :param chromosome the solution
    return standard solution
    '''

    result = []
    new_chrom = []
    for element in chromosome:
        temp = []
        for e in element:
            temp.append((e[0],e[1]))
        new_chrom.append(temp)

    for element in new_chrom:
        result.append(0)
        result.extend(element)
        result.append(0)
    return result



def gene_algorithm(size, process_num,time_limit,seed):

    '''
    This function the frame of the whole algorithm
    :param size is the scale of each generation
    :param process_num is the processing number
    :param time_limit is the time threshold
    :param seed is the given random seed
    return solution and corresponding cost
    '''

    global total_child_fit
    global total_child_pop
    global total_child_cost


    # initial population and other variables

    parent_population, parent_fitness, parent_totalcost = first_population(size)
    cost = min(parent_fitness)
    counter = 0
    result = [(cost, parent_population[parent_fitness.index(cost)])]
    penalty = 1
    percent = 0.5
    mutate_possib = 0.4
    current_best_fitness = min(parent_fitness)
    current_best_popluation = parent_population[parent_fitness.index(current_best_fitness)]

    while 1:

        # count for the frequency of the change of penalty parameter
        counter += 1
        # multiprocessing init
        margin = size/process_num
        process_i = 0

        # Restart algorithm based on the average, std and median
        if abs(numpy.average(parent_fitness)-numpy.median(parent_fitness))<0.05 or numpy.std(parent_fitness)<5:
            # print "restarting..."
            parent_population, parent_fitness, parent_totalcost = first_population(size)
            current_best_fitness = min(parent_fitness)
            current_best_popluation = parent_population[parent_fitness.index(current_best_fitness)]

        else:
            # random select a cross over function
            switch = False
            if counter>2:
                if random.uniform(0,1)<mutate_possib:
                    switch = True
                else:
                    switch = False
            # multithreading
            while process_i < process_num:
                parameter = (parent_population[process_i * margin:(process_i + 1) * margin],
                             parent_fitness[process_i * margin:(process_i + 1) * margin], size/process_num,
                             mutate_possib,capacity,dijstra_dict,depot_node,penalty,switch,seed)
                process_queue.put(parameter)
                process_i += 1
            muti_process(process_num, process_queue)
            process_queue.queue.clear()
            # filter the total generation

            parent_population, parent_fitness, parent_totalcost,current_best_popluation,current_best_fitness = shrink(total_child_pop,total_child_fit,total_child_cost,
                                                                                                                     parent_population,parent_fitness,parent_totalcost,size,percent)
        # clear the container for multi processing
        del total_child_pop[:]
        del total_child_fit[:]
        del total_child_cost[:]

        # update current best solution
        if current_best_fitness<cost:

            cost = current_best_fitness
            result.append((current_best_fitness,current_best_popluation))

        # update penalty parameter
        if counter%10 == 0:
            penalty += 1
            # print cost, "\t time elapse: ", time.time() - starttime, " s"
            # print "avg", numpy.average(parent_fitness), "mid", numpy.median(parent_fitness),"std",numpy.std(parent_fitness)
            # print '-----------------------------------------------'

        if time.time()-starttime>time_limit-1:
            min_cost = result[-1][0]
            solution = result[-1][1]
            break

    return min_cost, solution



if __name__ == "__main__":
    path = sys.argv
    file_name = path[1]
    time_limit = float(path[3])
    seed = path[5]
    # file_name = "ForStus/CARP_samples/new_gdb5.dat"
    # time_limit = 30.0
    # seed = 0
    random.seed(seed)
    population_size = 200
    process_number = 8
    read_file(file_name)
    get_total_mincost_dict()
    min_cost, chromosome = gene_algorithm(population_size,process_number,time_limit,seed)
    # sum = []
    # for ele in chromosome:
    #     sum += ele
    # print len(set(sum))
    print "s", (",".join(str(d) for d in to_format(chromosome))).replace(" ", "")
    print "q", (min_cost)