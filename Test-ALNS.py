import math
import random
import copy
import matplotlib.pyplot as plt
# --- Constants and Configuration ---
HUB_ID = 1
DEFAULT_SPEED = 350 # meters per minute (from Case Study Section 6.1)

# --- Helper Functions ---
def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_travel_time(p1, p2, speed=DEFAULT_SPEED):
    if speed == 0: return float('inf')
    dist = euclidean_distance(p1, p2)
    return dist / speed

#########################
# --- Data Structures ---
#########################
class Node:
    def __init__(self, id, node_type, x, y, service_time=0, early=0, latest=float('inf')):
        self.id = id
        self.node_type = node_type
        self.x = x
        self.y = y
        self.service_time = service_time
        self.early = early
        self.latest = latest

    def __repr__(self):
        return f"Node({self.id}, '{self.node_type}', X:{self.x}, Y:{self.y})"

class Hub(Node):
    def __init__(self, id, x, y):
        super().__init__(id, 'hub', x, y)

class Satellite(Node): # Represents a PHYSICAL Satellite
    def __init__(self, id, x, y, service_time):
        super().__init__(id, 'satellite', x, y, service_time=service_time)
        self.fe_arrival_time_dist = -1
        self.fe_departure_time_dist = -1
        self.fe_arrival_time_coll = -1
        self.fe_departure_time_coll = -1
        self.parcels_to_deliver_by_fe = {}
        self.parcels_to_pickup_by_fe = {}

class Customer(Node):
    def __init__(self, id, node_type, x, y, service_time, early, latest, demand, deadline=float('inf')):
        super().__init__(id, node_type, x, y, service_time, early, latest)
        self.demand = demand
        self.deadline_at_hub = deadline if node_type == 'pickup_customer' else float('inf')
        self.is_served = False
        self.assigned_se_route_id = None
        self.assigned_fe_route_id = None

    def __repr__(self):
        return f"Cust({self.id}, '{self.node_type}', D:{self.demand}, E:{self.early}, L:{self.latest})"

class SERoute:
    def __init__(self, id, physical_satellite_obj, vehicle_capacity):
        self.id = id
        self.physical_satellite = physical_satellite_obj
        self.vehicle_capacity = vehicle_capacity
        self.customer_sequence = []
        self.arrival_times = {}
        self.departure_times = {}
        self.load_at_arrival = {}
        self.load_at_departure = {}
        self.departure_from_dist_sat_time = 0
        self.arrival_at_coll_sat_time = 0
        self.current_total_delivery_demand_onboard = 0
        self.current_total_pickup_demand_collected = 0
        self.cost = 0

    def get_customers_by_type(self, cust_type_str):
        return [c for c in self.customer_sequence if c.node_type == cust_type_str]

    def calculate_metrics(self):
        if not self.customer_sequence:
            self.cost = 0
            self.departure_from_dist_sat_time = 0
            self.arrival_at_coll_sat_time = 0
            self.current_total_delivery_demand_onboard = 0
            self.current_total_pickup_demand_collected = 0
            self.arrival_times.clear(); self.departure_times.clear(); self.load_at_arrival.clear(); self.load_at_departure.clear()
            return True

        self.current_total_delivery_demand_onboard = sum(abs(c.demand) for c in self.get_customers_by_type('delivery_customer'))
        self.current_total_pickup_demand_collected = 0
        
        current_vehicle_load = self.current_total_delivery_demand_onboard
        if current_vehicle_load > self.vehicle_capacity:
            return False

        fe_supplies_ready_time = self.physical_satellite.fe_arrival_time_dist \
                                 if self.physical_satellite.fe_arrival_time_dist != -1 else 0.0

        current_time_at_node = 0.0
        if self.customer_sequence: # Check if there are any customers before accessing index 0
            first_cust = self.customer_sequence[0]
            tt_to_first = calculate_travel_time(self.physical_satellite, first_cust)
            earliest_arrival_at_first = max(first_cust.early, fe_supplies_ready_time + tt_to_first)
            
            if earliest_arrival_at_first > first_cust.latest + 1e-9 : return False
            current_time_at_node = earliest_arrival_at_first - tt_to_first
            self.departure_from_dist_sat_time = current_time_at_node
        else: # No customers
             self.departure_from_dist_sat_time = fe_supplies_ready_time


        route_cost = 0.0
        last_visited_node_obj = self.physical_satellite

        for i, cust in enumerate(self.customer_sequence):
            travel_t = calculate_travel_time(last_visited_node_obj, cust)
            arrival_at_cust = current_time_at_node + travel_t
            
            self.arrival_times[cust.id] = max(arrival_at_cust, cust.early)
            if self.arrival_times[cust.id] > cust.latest + 1e-9: return False

            self.load_at_arrival[cust.id] = current_vehicle_load
            
            service_start_time = self.arrival_times[cust.id]
            current_time_at_node = service_start_time + cust.service_time
            self.departure_times[cust.id] = current_time_at_node

            if cust.node_type == 'delivery_customer':
                current_vehicle_load -= abs(cust.demand)
            elif cust.node_type == 'pickup_customer':
                current_vehicle_load += cust.demand
                self.current_total_pickup_demand_collected += cust.demand
            
            self.load_at_departure[cust.id] = current_vehicle_load
            if current_vehicle_load > self.vehicle_capacity + 1e-9 or current_vehicle_load < -1e-9 : return False
            
            route_cost += euclidean_distance(last_visited_node_obj, cust)
            last_visited_node_obj = cust

        if self.customer_sequence:
            route_cost += euclidean_distance(last_visited_node_obj, self.physical_satellite)
            self.arrival_at_coll_sat_time = current_time_at_node + calculate_travel_time(last_visited_node_obj, self.physical_satellite)
        else:
            self.arrival_at_coll_sat_time = self.departure_from_dist_sat_time

        self.cost = route_cost
        return True

    def __repr__(self):
        return f"SERoute({self.id}, Sat:{self.physical_satellite.id}, Cost:{self.cost:.2f}, Custs:{len(self.customer_sequence)})"

class FERoute:
    def __init__(self, id, hub_obj, vehicle_capacity):
        self.id = id
        self.hub = hub_obj
        self.vehicle_capacity = vehicle_capacity
        self.conceptual_visits_sequence = []
        self.arrival_times_at_conceptual_visit = {}
        self.departure_times_from_conceptual_visit = {}
        self.load_after_leaving_conceptual_visit = {}
        self.delivery_parcels_onboard_state = {}
        self.pickup_parcels_onboard_state = {}
        self.cost = 0
        self.initial_deliveries_from_hub = {}
        self.parcels_serviced_by_visit = {}

    def get_conceptual_visit_id_str(self, physical_sat_id, role_str):
        return f"-{physical_sat_id}" if role_str == 'coll' else str(physical_sat_id)

    def calculate_metrics(self, all_customers_map):
        if not self.conceptual_visits_sequence:
            self.cost = 0
            hub_depart_key = (self.hub.id, 'depart_hub', -1)
            hub_arrival_key = (self.hub.id, 'arrive_hub', -1)
            self.delivery_parcels_onboard_state[hub_depart_key] = []
            self.pickup_parcels_onboard_state[hub_depart_key] = []
            self.delivery_parcels_onboard_state[hub_arrival_key] = []
            self.pickup_parcels_onboard_state[hub_arrival_key] = []
            self.load_after_leaving_conceptual_visit[hub_depart_key] = 0
            self.load_after_leaving_conceptual_visit[hub_arrival_key] = 0
            return True

        current_time = 0.0
        current_vehicle_load = 0.0
        
        onboard_delivery_parcels_fe = dict(self.initial_deliveries_from_hub)
        onboard_pickup_parcels_fe = {}
        
        for cust_id, demand in onboard_delivery_parcels_fe.items():
            current_vehicle_load += abs(demand)

        hub_depart_key = (self.hub.id, 'depart_hub', -1)
        self.load_after_leaving_conceptual_visit[hub_depart_key] = current_vehicle_load
        self.delivery_parcels_onboard_state[hub_depart_key] = sorted(list(onboard_delivery_parcels_fe.keys()))
        self.pickup_parcels_onboard_state[hub_depart_key] = []

        if current_vehicle_load > self.vehicle_capacity + 1e-9: return False

        current_node_obj = self.hub
        route_cost = 0.0
        
        # Ensure parcels_serviced_by_visit is initialized for all conceptual visits on this route
        for visit_idx_init, (phys_sat_obj_init, role_init) in enumerate(self.conceptual_visits_sequence):
            conceptual_visit_key_init = (phys_sat_obj_init.id, role_init, visit_idx_init)
            if conceptual_visit_key_init not in self.parcels_serviced_by_visit:
                 self.parcels_serviced_by_visit[conceptual_visit_key_init] = {}


        for visit_idx, (phys_sat_obj, role) in enumerate(self.conceptual_visits_sequence):
            conceptual_visit_key = (phys_sat_obj.id, role, visit_idx)
            # Initialize if not already (e.g. for 'dist' tasks not pre-filled by greedy construction)
            if conceptual_visit_key not in self.parcels_serviced_by_visit:
                self.parcels_serviced_by_visit[conceptual_visit_key] = {}


            travel_t = calculate_travel_time(current_node_obj, phys_sat_obj)
            arrival_at_sat = current_time + travel_t
            self.arrival_times_at_conceptual_visit[conceptual_visit_key] = arrival_at_sat
            service_start_time = arrival_at_sat
            
            if role == 'dist':
                delivered_here_ids = []
                for cust_id, demand_val in list(onboard_delivery_parcels_fe.items()):
                    if cust_id in phys_sat_obj.parcels_to_deliver_by_fe: # Check against satellite's total need
                        current_vehicle_load -= abs(demand_val)
                        self.parcels_serviced_by_visit[conceptual_visit_key][cust_id] = demand_val
                        delivered_here_ids.append(cust_id)
                        del onboard_delivery_parcels_fe[cust_id]
            
            elif role == 'coll':
                parcels_this_visit_should_pickup = self.parcels_serviced_by_visit.get(conceptual_visit_key, {})
                for cust_id, demand_val in parcels_this_visit_should_pickup.items():
                    if current_vehicle_load + abs(demand_val) <= self.vehicle_capacity + 1e-9 : # demand_val is positive
                        current_vehicle_load += abs(demand_val)
                        customer_obj = all_customers_map[cust_id]
                        onboard_pickup_parcels_fe[cust_id] = (abs(demand_val), customer_obj.deadline_at_hub)
                    else:
                        # This means the construction heuristic assigned more than capacity allows for this specific route config
                        # print(f"FE Route {self.id} capacity logic error during coll at {phys_sat_obj.id} for cust {cust_id}")
                        return False

            self.delivery_parcels_onboard_state[conceptual_visit_key] = sorted(list(onboard_delivery_parcels_fe.keys()))
            self.pickup_parcels_onboard_state[conceptual_visit_key] = sorted(list(onboard_pickup_parcels_fe.keys()))

            if current_vehicle_load < -1e-9 or current_vehicle_load > self.vehicle_capacity + 1e-9 :
                return False

            current_time = service_start_time + phys_sat_obj.service_time
            self.departure_times_from_conceptual_visit[conceptual_visit_key] = current_time
            self.load_after_leaving_conceptual_visit[conceptual_visit_key] = current_vehicle_load
            
            route_cost += euclidean_distance(current_node_obj, phys_sat_obj)
            current_node_obj = phys_sat_obj

        travel_to_hub_t = calculate_travel_time(current_node_obj, self.hub)
        arrival_at_hub_time = current_time + travel_to_hub_t
        route_cost += euclidean_distance(current_node_obj, self.hub)
        self.cost = route_cost
        
        hub_arrival_key = (self.hub.id, 'arrive_hub', -1)
        self.load_after_leaving_conceptual_visit[hub_arrival_key] = current_vehicle_load
        if onboard_delivery_parcels_fe:
            # print(f"Warning FE Route {self.id}: Undelivered parcels {list(onboard_delivery_parcels_fe.keys())} at hub arrival.")
            return False
        self.delivery_parcels_onboard_state[hub_arrival_key] = []
        self.pickup_parcels_onboard_state[hub_arrival_key] = sorted(list(onboard_pickup_parcels_fe.keys()))

        for cust_id, (demand, deadline) in onboard_pickup_parcels_fe.items():
            if arrival_at_hub_time > deadline + 1e-9:
                return False
        return True

    def __repr__(self):
        num_visits = len(self.conceptual_visits_sequence)
        return f"FERoute({self.id}, Cost:{self.cost:.2f}, ConceptualVisits:{num_visits})"


def parse_input(file_path):
    hub = None
    parsed_satellites_temp = []
    parsed_customers_temp = []
    final_satellites = []
    final_customers = []
    fe_capacity = 0.0 # Use float for capacities
    se_capacity = 0.0

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Normalize header: remove leading/trailing spaces, convert to lower case
        raw_headers = [h.strip() for h in lines[0].split(',')]
        headers = [h.lower().replace(' ', '_') for h in raw_headers] # e.g. "Service Time " -> "service_time_"
        # Further normalize specific known variations
        header_map_corrected = {}
        for i, h_norm in enumerate(headers):
            if "service_time" in h_norm : header_map_corrected['service_time'] = i
            elif "fe_cap" in h_norm : header_map_corrected['fe_cap'] = i
            elif "se_cap" in h_norm : header_map_corrected['se_cap'] = i
            else: header_map_corrected[h_norm.replace('_','')] = i # Remove underscore for simple keys like 'x', 'y'

        headers = header_map_corrected # Use the corrected map

        for line_num_original_file, line_content in enumerate(lines[1:]):
            values = [v.strip() for v in line_content.split(',')]
            
            def get_val(key_name, default_val=0.0, is_float=True): # Default to float for most numeric values
                idx = headers.get(key_name, -1)
                if idx != -1 and idx < len(values) and values[idx]:
                    try:
                        return float(values[idx]) if is_float else values[idx]
                    except ValueError:
                        # print(f"Warning: Could not parse '{values[idx]}' as float for '{key_name}' on line {line_num_original_file+2}. Using default {default_val}.")
                        return default_val
                return default_val

            node_type_val = int(get_val('type', default_val=0.0)) # Type is int
            x_coord = get_val('x')
            y_coord = get_val('y')
            serv_time = get_val('service_time', default_val=0.0)
            
            if node_type_val == 0: # Hub
                hub = Hub(id=HUB_ID, x=x_coord, y=y_coord) # Hub service time is 0
                fe_cap_val = get_val('fe_cap', default_val=0.0)
                se_cap_val = get_val('se_cap', default_val=0.0)
                if fe_cap_val > 1e-9: fe_capacity = fe_cap_val # Check against small epsilon
                if se_cap_val > 1e-9: se_capacity = se_cap_val
            else:
                e_tw = get_val('early', default_val=0.0)
                l_tw = get_val('latest', default_val=float('inf'))
                dem = get_val('demand', default_val=0.0)
                
                if node_type_val == 1: # Satellite
                    parsed_satellites_temp.append(
                        {'original_line': line_num_original_file,
                         'obj_data': {'x': x_coord, 'y': y_coord, 'service_time': serv_time}}
                    )
                elif node_type_val == 2: # Delivery Customer
                    cust_data = {'original_line': line_num_original_file, 'type_val': node_type_val,
                                 'obj_data': {'node_type': 'delivery_customer', 'x': x_coord, 'y': y_coord,
                                              'service_time': serv_time, 'early': e_tw, 'latest': l_tw,
                                              'demand': -abs(dem)}}
                    parsed_customers_temp.append(cust_data)
                elif node_type_val == 3: # Pickup Customer
                    deadln = get_val('deadline', default_val=float('inf'))
                    cust_data = {'original_line': line_num_original_file, 'type_val': node_type_val,
                                 'obj_data': {'node_type': 'pickup_customer', 'x': x_coord, 'y': y_coord,
                                              'service_time': serv_time, 'early': e_tw, 'latest': l_tw,
                                              'demand': abs(dem), 'deadline': deadln}}
                    parsed_customers_temp.append(cust_data)

    current_node_id_counter = HUB_ID + 1
    parsed_satellites_temp.sort(key=lambda x: x['original_line'])
    for sat_data_entry in parsed_satellites_temp:
        sd = sat_data_entry['obj_data']
        final_satellites.append(Satellite(id=current_node_id_counter, x=sd['x'], y=sd['y'], service_time=sd['service_time']))
        current_node_id_counter += 1

    parsed_customers_temp.sort(key=lambda x: (x['type_val'], x['original_line']))
    for cust_data_entry in parsed_customers_temp:
        cd = cust_data_entry['obj_data']
        final_customers.append(Customer(id=current_node_id_counter, **cd))
        current_node_id_counter += 1
    
    if (fe_capacity < 1e-9 or se_capacity < 1e-9) and len(lines) > 1:
        first_data_line_content = lines[1]
        first_data_line_values = [v.strip() for v in first_data_line_content.split(',')]
        if len(first_data_line_values) == len(raw_headers): # Use raw_headers count for consistency
            def get_val_fallback(key_name, default_val=0.0):
                idx = headers.get(key_name, -1) # Use corrected headers map
                if idx != -1 and idx < len(first_data_line_values) and first_data_line_values[idx]:
                    try: return float(first_data_line_values[idx])
                    except ValueError: return default_val
                return default_val
            if fe_capacity < 1e-9: fe_capacity = get_val_fallback('fe_cap', default_val=0.0)
            if se_capacity < 1e-9: se_capacity = get_val_fallback('se_cap', default_val=0.0)
            
    return hub, final_satellites, final_customers, fe_capacity, se_capacity


def build_initial_solution(hub, physical_satellites_list, customers_list, fe_cap, se_cap, all_customers_map_for_fe):
    solution_se_routes = []
    next_se_route_id = 201
    
    shuffled_customers = list(customers_list)
    random.shuffle(shuffled_customers)

    for cust_to_insert in shuffled_customers:
        if cust_to_insert.is_served: continue
        inserted_successfully = False
        if solution_se_routes:
            target_se_route = solution_se_routes[-1]
            original_seq = list(target_se_route.customer_sequence)
            original_cost = target_se_route.cost
            
            target_se_route.customer_sequence.append(cust_to_insert)
            if target_se_route.calculate_metrics():
                inserted_successfully = True
                cust_to_insert.is_served = True
                cust_to_insert.assigned_se_route_id = target_se_route.id
            else:
                target_se_route.customer_sequence = original_seq
                target_se_route.cost = original_cost
                if not target_se_route.calculate_metrics() and original_seq:
                    pass
        if not inserted_successfully:
            best_new_se_route = None
            min_cost_for_new_route = float('inf')
            for phys_sat_obj in physical_satellites_list:
                temp_new_se_route = SERoute(next_se_route_id, phys_sat_obj, se_cap)
                temp_new_se_route.customer_sequence = [cust_to_insert]
                if temp_new_se_route.calculate_metrics():
                    if temp_new_se_route.cost < min_cost_for_new_route:
                        min_cost_for_new_route = temp_new_se_route.cost
                        best_new_se_route = temp_new_se_route
            if best_new_se_route:
                solution_se_routes.append(best_new_se_route)
                cust_to_insert.is_served = True
                cust_to_insert.assigned_se_route_id = best_new_se_route.id
                next_se_route_id += 1

    for phys_sat_obj in physical_satellites_list:
        phys_sat_obj.parcels_to_deliver_by_fe.clear()
        phys_sat_obj.parcels_to_pickup_by_fe.clear()
    for se_route in solution_se_routes:
        phys_sat_for_se = se_route.physical_satellite
        for cust_in_se in se_route.customer_sequence:
            if cust_in_se.node_type == 'delivery_customer':
                phys_sat_for_se.parcels_to_deliver_by_fe[cust_in_se.id] = cust_in_se.demand
            elif cust_in_se.node_type == 'pickup_customer':
                phys_sat_for_se.parcels_to_pickup_by_fe[cust_in_se.id] = cust_in_se.demand
    
    solution_fe_routes = []
    next_fe_route_id = 11
    unfulfilled_fe_tasks = []
    for s_obj in physical_satellites_list:
        if s_obj.parcels_to_deliver_by_fe:
            unfulfilled_fe_tasks.append({'sat_obj': s_obj, 'role': 'dist', 'parcels': dict(s_obj.parcels_to_deliver_by_fe), 'original_total_demand': sum(abs(d) for d in s_obj.parcels_to_deliver_by_fe.values())})
        if s_obj.parcels_to_pickup_by_fe:
            unfulfilled_fe_tasks.append({'sat_obj': s_obj, 'role': 'coll', 'parcels': dict(s_obj.parcels_to_pickup_by_fe), 'original_total_demand': sum(abs(d) for d in s_obj.parcels_to_pickup_by_fe.values())})
    
    while any(task['parcels'] for task in unfulfilled_fe_tasks):
        current_fe_route = FERoute(next_fe_route_id, hub, fe_cap)
        current_fe_route.initial_deliveries_from_hub = {}
        
        can_add_more_to_this_fe_route = True
        while can_add_more_to_this_fe_route:
            best_task_to_add_info = None # (task_idx, cost_of_adding, parcels_serviced_this_visit_dict, temp_route_obj)
            
            for task_idx, current_task_data in enumerate(unfulfilled_fe_tasks):
                task_sat_obj, task_role, remaining_parcels_for_task_dict = current_task_data['sat_obj'], current_task_data['role'], current_task_data['parcels']
                if not remaining_parcels_for_task_dict: continue

                # Simulate adding this task (or part of it)
                temp_route_for_eval = copy.deepcopy(current_fe_route)
                temp_route_for_eval.conceptual_visits_sequence.append((task_sat_obj, task_role))
                
                # Tentatively decide what this visit will service
                parcels_for_this_visit_attempt = {}
                current_load_for_visit_attempt = 0

                if task_role == 'dist':
                    # Try to load all remaining parcels for this dist task, ensure they are in initial_deliveries
                    temp_initial_deliveries = dict(temp_route_for_eval.initial_deliveries_from_hub)
                    for cust_id, dem_val in remaining_parcels_for_task_dict.items():
                        parcels_for_this_visit_attempt[cust_id] = dem_val # Assume we attempt to deliver all remaining
                        if cust_id not in temp_initial_deliveries:
                             temp_initial_deliveries[cust_id] = dem_val
                    temp_route_for_eval.initial_deliveries_from_hub = temp_initial_deliveries
                
                elif task_role == 'coll':
                    # For coll, calculate_metrics will figure out what can be picked up based on current load and capacity.
                    # We pass the *available* parcels for pickup at this conceptual visit.
                    conceptual_visit_key_for_pickup = (task_sat_obj.id, task_role, len(temp_route_for_eval.conceptual_visits_sequence)-1)
                    temp_route_for_eval.parcels_serviced_by_visit[conceptual_visit_key_for_pickup] = dict(remaining_parcels_for_task_dict)


                if temp_route_for_eval.calculate_metrics(all_customers_map_for_fe):
                    cost_of_adding_this_visit = temp_route_for_eval.cost - current_fe_route.cost
                    
                    last_added_visit_key = (task_sat_obj.id, task_role, len(temp_route_for_eval.conceptual_visits_sequence)-1)
                    parcels_actually_serviced_in_last_visit = temp_route_for_eval.parcels_serviced_by_visit.get(last_added_visit_key, {})

                    if parcels_actually_serviced_in_last_visit: # If the visit did something
                        if best_task_to_add_info is None or cost_of_adding_this_visit < best_task_to_add_info[1]:
                            best_task_to_add_info = (task_idx, cost_of_adding_this_visit, parcels_actually_serviced_in_last_visit, temp_route_for_eval)
            
            if best_task_to_add_info:
                added_task_original_idx, _, serviced_parcels_dict, adopted_fe_route_state = best_task_to_add_info
                current_fe_route = adopted_fe_route_state # Adopt the successful state

                # Update the original unfulfilled_fe_tasks list
                parcels_dict_in_task_list = unfulfilled_fe_tasks[added_task_original_idx]['parcels']
                for cust_id_serviced, _ in serviced_parcels_dict.items():
                    if cust_id_serviced in parcels_dict_in_task_list:
                        del parcels_dict_in_task_list[cust_id_serviced]
            else:
                can_add_more_to_this_fe_route = False

        if current_fe_route.conceptual_visits_sequence:
            # One final validation of the fully constructed route
            if current_fe_route.calculate_metrics(all_customers_map_for_fe):
                 solution_fe_routes.append(current_fe_route)
                 next_fe_route_id += 1
            # else:
                 # This implies a logic flaw if individual additions were feasible but final route isn't.
                 # Or, tasks it thought it serviced need rollback (very complex for initial).
                 # print(f"Error: Final FE route {current_fe_route.id} failed validation. Data might be inconsistent.")
                 # For simplicity, if this happens, the tasks it "claimed" are still marked as serviced.
                 pass
                 
    # Synchronization Update
    for fe_r in solution_fe_routes:
        for visit_idx, (phys_sat_obj, role) in enumerate(fe_r.conceptual_visits_sequence):
            conceptual_visit_key = (phys_sat_obj.id, role, visit_idx)
            arrival = fe_r.arrival_times_at_conceptual_visit.get(conceptual_visit_key, -1)
            departure = fe_r.departure_times_from_conceptual_visit.get(conceptual_visit_key, -1)

            if arrival == -1 or departure == -1: continue # Skip if times not set

            if role == 'dist':
                if phys_sat_obj.fe_arrival_time_dist == -1 or arrival < phys_sat_obj.fe_arrival_time_dist :
                     phys_sat_obj.fe_arrival_time_dist = arrival
                if phys_sat_obj.fe_departure_time_dist == -1 or departure > phys_sat_obj.fe_departure_time_dist :
                     phys_sat_obj.fe_departure_time_dist = departure
            elif role == 'coll':
                if phys_sat_obj.fe_arrival_time_coll == -1 or arrival < phys_sat_obj.fe_arrival_time_coll :
                     phys_sat_obj.fe_arrival_time_coll = arrival
                if phys_sat_obj.fe_departure_time_coll == -1 or departure > phys_sat_obj.fe_departure_time_coll :
                     phys_sat_obj.fe_departure_time_coll = departure
    
    # Re-calculate SE routes to ensure their timings are consistent with FE satellite visit times
    # This is important as SE departure for deliveries depends on FE arrival at satellite.
    final_feasible_se_routes = []
    for se_r in solution_se_routes:
        if se_r.calculate_metrics(): # Recalculate with updated satellite fe_arrival_time_dist
            final_feasible_se_routes.append(se_r)
        # else:
            # print(f"Warning: SE Route {se_r.id} (Sat {se_r.physical_satellite.id}) became infeasible after FE synchronization. Discarding.")
            # Unserve customers on this SE route
            # for cust_on_failed_se in se_r.customer_sequence:
            #     cust_on_failed_se.is_served = False
            #     cust_on_failed_se.assigned_se_route_id = None
                # Potentially, FE routes might also need adjustment if SE routes are discarded.
                # This is beyond the scope of a simple initial heuristic.

    return solution_fe_routes, final_feasible_se_routes


def format_solution_output(fe_routes, se_routes, customers_map_for_output, hub_obj, physical_satellites_map_ignored):
    output_lines = []
    total_cost = sum(r.cost for r in fe_routes) + sum(r.cost for r in se_routes)
    output_lines.append(f"Total Cost: {total_cost:.2f}")

    for fe_route in fe_routes:
        if not fe_route.conceptual_visits_sequence and fe_route.cost == 0: continue

        output_lines.append(f"Vehicle: {fe_route.id}  Cost: {fe_route.cost:.2f}")
        
        route_str_nodes = [str(hub_obj.id)]
        for visit_idx, (phys_sat_obj, role) in enumerate(fe_route.conceptual_visits_sequence):
            display_id = fe_route.get_conceptual_visit_id_str(phys_sat_obj.id, role)
            route_str_nodes.append(display_id)
        route_str_nodes.append(str(hub_obj.id))
        output_lines.append(f"Route: {' '.join(route_str_nodes)}")
        
        hub_depart_key = (hub_obj.id, 'depart_hub', -1)
        hub_depart_load = fe_route.load_after_leaving_conceptual_visit.get(hub_depart_key, 0)
        hub_delivery_parcels_str = ",".join(map(str, fe_route.delivery_parcels_onboard_state.get(hub_depart_key, [])))
        output_lines.append(f" {hub_obj.id} [ {hub_depart_load:.0f}   0:  0 [{hub_delivery_parcels_str}]]")

        for visit_idx, (phys_sat_obj, role) in enumerate(fe_route.conceptual_visits_sequence):
            conceptual_visit_key = (phys_sat_obj.id, role, visit_idx)
            display_id_detail = fe_route.get_conceptual_visit_id_str(phys_sat_obj.id, role)
            
            load_val = fe_route.load_after_leaving_conceptual_visit.get(conceptual_visit_key, 0)
            arrival = fe_route.arrival_times_at_conceptual_visit.get(conceptual_visit_key, 0)
            departure = fe_route.departure_times_from_conceptual_visit.get(conceptual_visit_key, 0)
            parcels_list_str = ",".join(map(str, fe_route.pickup_parcels_onboard_state.get(conceptual_visit_key, [])))
            output_lines.append(f" {display_id_detail} [ {load_val:.0f}  {arrival:.0f}:{departure:.0f} [{parcels_list_str}]]")

        hub_arrival_key = (hub_obj.id, 'arrive_hub', -1)
        hub_arrival_load = fe_route.load_after_leaving_conceptual_visit.get(hub_arrival_key, 0)
        hub_arrival_time = 0.0
        if fe_route.conceptual_visits_sequence:
            last_phys_sat_obj, _ = fe_route.conceptual_visits_sequence[-1]
            last_idx = len(fe_route.conceptual_visits_sequence)-1
            last_conceptual_visit_key = (last_phys_sat_obj.id, fe_route.conceptual_visits_sequence[-1][1], last_idx)
            last_sat_departure_time = fe_route.departure_times_from_conceptual_visit.get(last_conceptual_visit_key)
            if last_sat_departure_time is not None:
                 hub_arrival_time = last_sat_departure_time + calculate_travel_time(last_phys_sat_obj, hub_obj)
        
        parcels_at_hub_arrival_str = ",".join(map(str, fe_route.pickup_parcels_onboard_state.get(hub_arrival_key,[])))
        output_lines.append(f" {hub_obj.id} [ {hub_arrival_load:.0f} {hub_arrival_time:.0f}:{hub_arrival_time:.0f} [{parcels_at_hub_arrival_str}]]")
        output_lines.append("")

    for se_route in se_routes:
        if not se_route.customer_sequence and se_route.cost == 0 : continue
        output_lines.append(f"Vehicle: {se_route.id}  Cost: {se_route.cost:.2f}")
        
        route_str_nodes_se = [str(se_route.physical_satellite.id)]
        route_str_nodes_se.extend([str(c.id) for c in se_route.customer_sequence])
        end_node_display_se_route_line = f"-{se_route.physical_satellite.id}"
        route_str_nodes_se.append(end_node_display_se_route_line)
        output_lines.append(f"Route: {' '.join(route_str_nodes_se)}")
        
        initial_se_load = sum(abs(c.demand) for c in se_route.get_customers_by_type('delivery_customer'))
        fe_delivery_arrival_time_at_sat_for_se = se_route.physical_satellite.fe_arrival_time_dist \
                                                 if se_route.physical_satellite.fe_arrival_time_dist != -1 else 0.0
        actual_se_departure_time = se_route.departure_from_dist_sat_time
        output_lines.append(f"  {se_route.physical_satellite.id} [ {initial_se_load:.0f}  {fe_delivery_arrival_time_at_sat_for_se:.0f}:{actual_se_departure_time:.0f} []]")
        
        for cust in se_route.customer_sequence:
            load_at_dep = se_route.load_at_departure.get(cust.id, 0)
            arrival_at_c = se_route.arrival_times.get(cust.id, 0)
            departure_from_c = se_route.departure_times.get(cust.id, 0)
            output_lines.append(f"  {cust.id} [ {load_at_dep:.0f}  {arrival_at_c:.0f}:{departure_from_c:.0f}]")

        final_se_load_at_sat_arrival = sum(c.demand for c in se_route.get_customers_by_type('pickup_customer'))
        arrival_time_back_at_sat_se = se_route.arrival_at_coll_sat_time
        se_collection_point_display_id_detail = f"-{se_route.physical_satellite.id}"
        output_lines.append(f"  {se_collection_point_display_id_detail} [ {final_se_load_at_sat_arrival:.0f} {arrival_time_back_at_sat_se:.0f}:{arrival_time_back_at_sat_se:.0f} []]")
        output_lines.append("")
        
    return "\n".join(output_lines)


def plot_routes(hub, satellites, customers, fe_routes, se_routes, title="Routes"):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 8))
    # Plot hub
    plt.scatter([hub.x], [hub.y], c='red', marker='*', s=200, label='Hub')

    # Plot satellites
    sat_x = [s.x for s in satellites]
    sat_y = [s.y for s in satellites]
    plt.scatter(sat_x, sat_y, c='orange', marker='s', s=120, label='Satellite')

    # Plot customers by type
    deliv_x = [c.x for c in customers if c.node_type == 'delivery_customer']
    deliv_y = [c.y for c in customers if c.node_type == 'delivery_customer']
    pick_x = [c.x for c in customers if c.node_type == 'pickup_customer']
    pick_y = [c.y for c in customers if c.node_type == 'pickup_customer']
    plt.scatter(deliv_x, deliv_y, c='blue', marker='o', s=60, label='Delivery Customer')
    plt.scatter(pick_x, pick_y, c='green', marker='^', s=60, label='Pickup Customer')

    # Plot FE routes (distribution: hub->satellite, collection: satellite->hub)
    for fe_route in fe_routes:
        if not fe_route.conceptual_visits_sequence:
            continue
        points = [hub] + [sat for sat, _ in fe_route.conceptual_visits_sequence] + [hub]
        for i, (sat, role) in enumerate(fe_route.conceptual_visits_sequence):
            if role == 'dist':
                plt.plot([hub.x, sat.x], [hub.y, sat.y], 'r--', alpha=0.5, linewidth=2, label='FE Distribution' if i == 0 else "")
            elif role == 'coll':
                plt.plot([sat.x, hub.x], [sat.y, hub.y], 'g--', alpha=0.5, linewidth=2, label='FE Collection' if i == 0 else "")

    # Plot SE routes (satellite to customers and back)
    for se_route in se_routes:
        if not se_route.customer_sequence:
            continue
        sat = se_route.physical_satellite
        seq = [sat] + se_route.customer_sequence + [sat]
        xs = [n.x for n in seq]
        ys = [n.y for n in seq]
        plt.plot(xs, ys, '-', color='black', alpha=0.6, linewidth=1.5, label='SE Route' if se_route == se_routes[0] else "")

    plt.legend(loc='best')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_file = "E:/LOG des/Documents/2e-vrp-pdd-main/2e-vrp-pdd-main/instance-set-2e-vrp-pdd/50 customers/A_50_5_D.csv" # Use the provided filename
    run_no = random.randint(1000,9999)
    
    hub, physical_satellites_list, customers_list, fe_capacity, se_capacity = parse_input(input_file)
    customers_map = {c.id: c for c in customers_list}

    print(f"Parsed Data: Hub ID: {hub.id if hub else 'N/A'}, #Satellites: {len(physical_satellites_list)}, #Customers: {len(customers_list)}")
    print(f"FE Capacity: {fe_capacity}, SE Capacity: {se_capacity}")

    if not hub or not customers_list or fe_capacity < 1e-9 or se_capacity < 1e-9:
        print("Error: Essential data not loaded correctly. Exiting.")
        # print(f"Hub: {hub}, Customers: {len(customers_list)}, FE Cap: {fe_capacity}, SE Cap: {se_capacity}")

    else:
        solution_fe_routes, solution_se_routes = \
            build_initial_solution(hub, physical_satellites_list, customers_list, fe_capacity, se_capacity, customers_map)

        print(f"\nInitial Solution Built:")
        print(f"  #FE Routes: {len(solution_fe_routes)}")
        print(f"  #SE Routes: {len(solution_se_routes)}")
        
        unserved_count = sum(1 for c_obj in customers_list if not c_obj.is_served)
        print(f"  Unserved customers by initial heuristic: {unserved_count}")

        output_str = format_solution_output(solution_fe_routes, solution_se_routes, customers_map, hub, {})
        
        base_input_file_name = input_file.split('/')[-1].replace('.csv', '')
        output_file_name = f"initial_solution_{base_input_file_name}_run{run_no}.txt"
        
        with open(output_file_name, "w") as f_out:
            f_out.write(output_str)
        print(f"\nFormatted output written to {output_file_name}")

    # Plot initial solution routes
    plot_routes(hub, physical_satellites_list, customers_list, solution_fe_routes, solution_se_routes, title="Initial Solution Routes")


# File: run_alns_2evrp.py

import math
import random # Keep for parts of code(4).py if not fully converted
import numpy.random as rnd
import copy
import time

# Assuming ALNS library files (ALNS.py, State.py, etc.) are in the same directory or accessible
from alns.criteria import SimulatedAnnealing
from alns import ALNS
from alns.State import State
from alns.weight_schemes.SimpleWeights import SimpleWeights # From alns.weight_schemes
# from alns.criteria import HillClimbing # Example, we'll make our own SA
# For AcceptanceCriterion base:
# If it's not directly importable, define a simple base class:
try:
    from alns.criteria import AcceptanceCriterion # If ALNS package has it
except ImportError:
    class AcceptanceCriterion: # Minimal stub if not found
        def __call__(self, rnd_state, best, current, candidate) -> bool:
            raise NotImplementedError


# Import everything from your code(4).py
# For this example, I'll assume its contents are available.
# If code(4).py is a module: from code_4 import *
# Otherwise, you might need to merge relevant parts or refactor.
# For now, let's assume the classes from code(4).py are defined in this scope or imported.
# --- Start of code(4).py content (or import) ---
# (Paste the content of code(4).py here, or ensure it's properly imported)
# For brevity, I will assume the classes Node, Hub, Satellite, Customer, SERoute, FERoute,
# parse_input, build_initial_solution, format_solution_output, euclidean_distance, calculate_travel_time
# are defined and available.
# Make sure to adjust `random` calls in `code(4).py` to use `rnd_state` if ALNS passes it.
# For instance, in `build_initial_solution`:
# def build_initial_solution(..., rnd_state_np=None):
#    if rnd_state_np:
#        shuffled_customers = list(customers_list)
#        rnd_state_np.shuffle(shuffled_customers)
#    else: # fallback to standard random
#        shuffled_customers = list(customers_list)
#        random.shuffle(shuffled_customers)


# --- End of code(4).py content (or import) ---

# --- Constants for ALNS ---
# These would typically come from the paper or tuning
DEGREE_OF_DESTRUCTION = 0.15  # Example: remove 15% of customers
SIMULATED_ANNEALING_START_TEMP = 5000
SIMULATED_ANNEALING_END_TEMP = 1
# SIMULATED_ANNEALING_STEP = 0.999 # Cooling rate if exponential
# Instead of step, let's use cooling_rate for exponential, or iterations for linear
COOLING_RATE = 0.9995 # For exponential cooling per iteration

# Operator scores for SimpleWeights [best, better, accepted, rejected]
# Values from Pisinger & Røpke (2007) for VRP:
# Reaction factor (op_decay in SimpleWeights) r = 0.1
# sigma1 (new global best) = 33
# sigma2 (better than current) = 9
# sigma3 (accepted) = 13
# (Rejected score is implicitly 0 if not set by a reward)
# The ALNS library's SimpleWeights adds (1-decay)*score.
# The scores in ALNS library are directly the rewards.
REACTION_FACTOR = 0.1 # op_decay
SCORES = [33, 9, 13, 0] # Corresponds to ALNS lib: best, better, accepted, rejected (rejected gets 0 reward usually)


class SolutionState(State):
    def __init__(self, hub, satellites_list, customers_list, customers_map,
                 fe_capacity, se_capacity, fe_routes, se_routes,
                 unserved_customers=None, rnd_state_np=None):
        self.hub = hub
        self.satellites_list = satellites_list
        self.customers_list = customers_list # Original full list
        self.customers_map = customers_map   # Original full map
        self.fe_capacity = fe_capacity
        self.se_capacity = se_capacity

        self.fe_routes = fe_routes
        self.se_routes = se_routes
        
        # unserved_customers: list of Customer objects not in any SE route
        self.unserved_customers = unserved_customers if unserved_customers is not None else []
        
        self.rnd_state_np = rnd_state_np if rnd_state_np is not None else rnd.RandomState()

        self.cached_objective = -1
        self.is_feasible = False # Will be set by recalculate
        self.recalculate_all_metrics_and_check_feasibility()


    def objective(self) -> float:
        if not self.is_feasible: # If recalculate marked it as infeasible
            # Penalize heavily for unserved customers if any exist after repair attempt
            # The penalty should be larger than any possible cost improvement
            penalty = len(self.unserved_customers) * 100000 # Large penalty per unserved customer
            return float('inf') if penalty > 0 else self.cached_objective # if no unserved, but other infeas
        return self.cached_objective

    def copy(self):
        # Deepcopy routes, shallow copy for hub, satellites, customers (as they are mostly static definitions)
        # Unserved customers list needs to be copied.
        return SolutionState(self.hub,
                             self.satellites_list, # These are refs to original objects
                             self.customers_list,
                             self.customers_map,
                             self.fe_capacity,
                             self.se_capacity,
                             copy.deepcopy(self.fe_routes),
                             copy.deepcopy(self.se_routes),
                             list(self.unserved_customers), # Copy the list of unserved customer objects
                             self.rnd_state_np
                            )
    
    def _get_current_satellite_demands(self):
        """ Helper to update satellite parcel lists based on current SE routes """
        for sat in self.satellites_list:
            sat.parcels_to_deliver_by_fe.clear()
            sat.parcels_to_pickup_by_fe.clear()

        for se_route in self.se_routes:
            if not se_route.customer_sequence: continue
            phys_sat_for_se = se_route.physical_satellite # This must be a ref to one in self.satellites_list
            for cust_in_se in se_route.customer_sequence:
                if cust_in_se.node_type == 'delivery_customer':
                    phys_sat_for_se.parcels_to_deliver_by_fe[cust_in_se.id] = cust_in_se.demand
                elif cust_in_se.node_type == 'pickup_customer':
                    phys_sat_for_se.parcels_to_pickup_by_fe[cust_in_se.id] = cust_in_se.demand
    
    def _rebuild_fe_routes_from_satellite_demands(self):
        """
        Rebuilds FE routes based on current demands at satellites.
        This adapts the FE construction logic from `build_initial_solution`.
        """
        # This is a simplified version. A more robust one would reuse FE vehicle IDs if possible.
        # For now, create new FE routes.
        self.fe_routes.clear() 
        
        next_fe_route_id = 11 # Reset IDs or manage them globally
        
        # Collect all FE tasks (distribution to satellites, collection from satellites)
        unfulfilled_fe_tasks = []
        for s_obj in self.satellites_list:
            if s_obj.parcels_to_deliver_by_fe:
                unfulfilled_fe_tasks.append({
                    'sat_obj': s_obj, 'role': 'dist', 
                    'parcels': dict(s_obj.parcels_to_deliver_by_fe),
                    'original_total_demand': sum(abs(d) for d in s_obj.parcels_to_deliver_by_fe.values())
                })
            if s_obj.parcels_to_pickup_by_fe:
                unfulfilled_fe_tasks.append({
                    'sat_obj': s_obj, 'role': 'coll', 
                    'parcels': dict(s_obj.parcels_to_pickup_by_fe),
                    'original_total_demand': sum(abs(d) for d in s_obj.parcels_to_pickup_by_fe.values())
                })
        
        # Sort tasks (e.g., by earliest deadline for pickups, or largest demand for dist) - optional
        # The original initial solution shuffles or processes them as found. Let's stick to that simplicity for now.

        temp_fe_routes = []
        while any(task['parcels'] for task in unfulfilled_fe_tasks):
            current_fe_route = FERoute(next_fe_route_id, self.hub, self.fe_capacity)
            current_fe_route.initial_deliveries_from_hub = {} 
            # parcels_serviced_by_visit is critical for FERoute.calculate_metrics
            current_fe_route.parcels_serviced_by_visit = {}


            can_add_more_to_this_fe_route = True
            while can_add_more_to_this_fe_route:
                best_task_to_add_info = None # (task_idx, cost_increase, parcels_serviced_this_visit, temp_route_obj)
                
                for task_idx, current_task_data in enumerate(unfulfilled_fe_tasks):
                    task_sat_obj, task_role = current_task_data['sat_obj'], current_task_data['role']
                    remaining_parcels_for_task_dict = current_task_data['parcels']
                    if not remaining_parcels_for_task_dict: continue

                    temp_route_for_eval = copy.deepcopy(current_fe_route) # Test adding to this route
                    temp_route_for_eval.conceptual_visits_sequence.append((task_sat_obj, task_role))
                    
                    # Tentatively decide what this visit will service
                    # This is where parcels_serviced_by_visit gets populated for the new visit
                    conceptual_visit_key_for_new_visit = (task_sat_obj.id, task_role, len(temp_route_for_eval.conceptual_visits_sequence)-1)
                    
                    # For 'dist', this new visit will service all remaining parcels for this task_sat_obj
                    # For 'coll', this new visit will *attempt* to service all remaining parcels for this task_sat_obj
                    temp_route_for_eval.parcels_serviced_by_visit[conceptual_visit_key_for_new_visit] = dict(remaining_parcels_for_task_dict)

                    # Populate initial_deliveries_from_hub based on *all* dist tasks in parcels_serviced_by_visit
                    temp_route_for_eval.initial_deliveries_from_hub.clear()
                    for cv_key, p_dict in temp_route_for_eval.parcels_serviced_by_visit.items():
                        s_id, r, v_idx = cv_key
                        if r == 'dist':
                            for c_id, dem in p_dict.items():
                                temp_route_for_eval.initial_deliveries_from_hub[c_id] = dem
                    
                    if temp_route_for_eval.calculate_metrics(self.customers_map):
                        cost_of_adding = temp_route_for_eval.cost - current_fe_route.cost # Approx cost increase
                        parcels_actually_serviced_in_last_visit = temp_route_for_eval.parcels_serviced_by_visit.get(conceptual_visit_key_for_new_visit, {})
                        
                        if parcels_actually_serviced_in_last_visit:
                            if best_task_to_add_info is None or cost_of_adding < best_task_to_add_info[1]:
                                best_task_to_add_info = (task_idx, cost_of_adding, parcels_actually_serviced_in_last_visit, temp_route_for_eval)
                
                if best_task_to_add_info:
                    added_task_original_idx, _, serviced_parcels_dict, adopted_fe_route_state = best_task_to_add_info
                    current_fe_route = adopted_fe_route_state # Adopt successful state

                    # Update unfulfilled_fe_tasks based on serviced_parcels_dict
                    task_definition_to_update = unfulfilled_fe_tasks[added_task_original_idx]
                    for cust_id_serviced, _ in serviced_parcels_dict.items():
                        if cust_id_serviced in task_definition_to_update['parcels']:
                            del task_definition_to_update['parcels'][cust_id_serviced]
                else:
                    can_add_more_to_this_fe_route = False

            if current_fe_route.conceptual_visits_sequence:
                # Final validation of the built route
                current_fe_route.initial_deliveries_from_hub.clear() # Repopulate for final check
                for cv_key, p_dict in current_fe_route.parcels_serviced_by_visit.items():
                    s_id, r, v_idx = cv_key
                    if r == 'dist':
                        for c_id, dem in p_dict.items():
                            current_fe_route.initial_deliveries_from_hub[c_id] = dem

                if current_fe_route.calculate_metrics(self.customers_map):
                    temp_fe_routes.append(current_fe_route)
                    next_fe_route_id += 1
                # else:
                    # This FE route, even if partially built, is infeasible.
                    # Parcels it "thought" it served are still in unfulfilled_fe_tasks (or should be rolled back).
                    # This simplistic greedy builder might leave tasks unserved if a route fails.
                    # For ALNS, if any FE task remains unfulfilled, the solution is infeasible.
                    # This part needs to be robust. If tasks remain, it's an issue.
                    # A simple fallback: if tasks remain after this loop, the whole state is infeasible.
                    pass # This could be a source of problems if not all tasks get scheduled.

        self.fe_routes = temp_fe_routes
        
        # Check if all FE tasks were actually covered
        for task_data in unfulfilled_fe_tasks:
            if task_data['parcels']: # If any parcel dict is not empty
                # print(f"Warning: FE Rebuild failed to fulfill all satellite tasks. Sat {task_data['sat_obj'].id}, Role {task_data['role']}")
                return False # FE rebuild failed to cover all needs
        return True


    def _update_satellite_fe_times_from_fe_routes(self):
        """ Helper to update satellite FE interaction times from calculated FE routes """
        for sat in self.satellites_list: # Reset times first
            sat.fe_arrival_time_dist = -1
            sat.fe_departure_time_dist = -1
            sat.fe_arrival_time_coll = -1
            sat.fe_departure_time_coll = -1

        for fe_r in self.fe_routes:
            for visit_idx, (phys_sat_obj, role) in enumerate(fe_r.conceptual_visits_sequence):
                conceptual_visit_key = (phys_sat_obj.id, role, visit_idx)
                arrival = fe_r.arrival_times_at_conceptual_visit.get(conceptual_visit_key, -1)
                departure = fe_r.departure_times_from_conceptual_visit.get(conceptual_visit_key, -1)

                if arrival == -1 or departure == -1: continue

                # Find the actual satellite object from self.satellites_list to update
                # This assumes phys_sat_obj in fe_r.conceptual_visits_sequence are identified by id
                # and we need to update the main satellite objects in self.satellites_list
                target_sat_obj = next((s for s in self.satellites_list if s.id == phys_sat_obj.id), None)
                if not target_sat_obj: continue


                if role == 'dist':
                    if target_sat_obj.fe_arrival_time_dist == -1 or arrival < target_sat_obj.fe_arrival_time_dist:
                        target_sat_obj.fe_arrival_time_dist = arrival
                    if target_sat_obj.fe_departure_time_dist == -1 or departure > target_sat_obj.fe_departure_time_dist:
                        target_sat_obj.fe_departure_time_dist = departure
                elif role == 'coll':
                    if target_sat_obj.fe_arrival_time_coll == -1 or arrival < target_sat_obj.fe_arrival_time_coll:
                        target_sat_obj.fe_arrival_time_coll = arrival
                    if target_sat_obj.fe_departure_time_coll == -1 or departure > target_sat_obj.fe_departure_time_coll:
                        target_sat_obj.fe_departure_time_coll = departure

    def recalculate_all_metrics_and_check_feasibility(self):
        self.is_feasible = False # Assume infeasible until proven otherwise
        
        # 0. Ensure all customers initially considered "served" are in routes, and "unserved" are not.
        # This should be maintained by operators.

        # 1. Update satellite demands based on current SE routes and unserved customers
        self._get_current_satellite_demands()

        # 2. Rebuild/Revalidate FE routes to serve these satellite demands
        # This is the most complex part. Using a simplified greedy rebuild for now.
        if not self._rebuild_fe_routes_from_satellite_demands():
            self.cached_objective = float('inf') # FE rebuild failed
            # print("DEBUG: FE Rebuild failed.")
            return # Infeasible state

        current_total_fe_cost = sum(r.cost for r in self.fe_routes)

        # 3. Update satellite FE arrival/departure times from the (new) FE routes
        self._update_satellite_fe_times_from_fe_routes()

        # 4. Recalculate all SE routes' metrics
        current_total_se_cost = 0
        all_se_feasible = True
        temp_active_se_routes = []
        for se_route in self.se_routes:
            if not se_route.customer_sequence: # Remove empty SE routes
                 # print(f"DEBUG: SE Route {se_route.id} is empty, removing.")
                 continue 
            
            # Ensure the physical_satellite object in se_route is the one from self.satellites_list
            # This is important if routes were deepcopied without care for shared satellite objects.
            # Best practice: SE routes store satellite ID, and get the object from the state's list.
            # For now, assume se_route.physical_satellite is correctly referenced or its FE times were updated.
            # The update in _update_satellite_fe_times_from_fe_routes modifies satellites in self.satellites_list.
            # If se_route.physical_satellite is a deep copy, it won't see these updates.
            # Let's fix this:
            master_sat_obj = next((s for s in self.satellites_list if s.id == se_route.physical_satellite.id), None)
            if master_sat_obj:
                se_route.physical_satellite = master_sat_obj # Point to the master satellite object
            else: # Should not happen
                all_se_feasible = False; break

            if not se_route.calculate_metrics():
                all_se_feasible = False
                # print(f"DEBUG: SE Route {se_route.id} (Sat {se_route.physical_satellite.id}) failed calc_metrics.")
                break
            current_total_se_cost += se_route.cost
            temp_active_se_routes.append(se_route)
        
        self.se_routes = temp_active_se_routes # Update SE routes list (removed empty ones)

        if not all_se_feasible:
            self.cached_objective = float('inf')
            # print("DEBUG: Not all SE routes feasible.")
            return # Infeasible state

        # 5. Final Feasibility Checks
        # Are all *original* customers (excluding those explicitly in self.unserved_customers) served?
        served_customer_ids_in_routes = set()
        for se_route in self.se_routes:
            for cust in se_route.customer_sequence:
                served_customer_ids_in_routes.add(cust.id)
        
        for cust in self.customers_list:
            if cust.id not in served_customer_ids_in_routes and cust not in self.unserved_customers:
                # This means a customer was dropped somehow and not marked unserved.
                # This indicates a logic error in operators or recalculation.
                # print(f"Error: Customer {cust.id} is lost (not served, not in unserved_customers list).")
                self.cached_objective = float('inf')
                return

        if self.unserved_customers: # If, after repair, some customers are still unserved
            self.cached_objective = float('inf') # Or add penalty as in self.objective()
            # print(f"DEBUG: {len(self.unserved_customers)} customers remain unserved.")
            return # Solution is infeasible for ALNS if it cannot serve all.

        self.cached_objective = current_total_fe_cost + current_total_se_cost
        self.is_feasible = True
        # print(f"DEBUG: Recalculation successful. Cost: {self.cached_objective}")


class SimulatedAnnealing(AcceptanceCriterion):
    def __init__(self, start_temp: float, end_temp: float, total_alns_iterations: int, cooling_rate: float = 0.995):
        if not (0 < cooling_rate <= 1):
            raise ValueError("Cooling rate must be in (0, 1]")
        if start_temp < end_temp or start_temp < 0 or end_temp < 0:
            raise ValueError("Temperatures must be non-negative with start_temp >= end_temp.")
        if total_alns_iterations <=0:
            raise ValueError("Total ALNS iterations must be positive.")

        self.start_temp = start_temp
        self.end_temp = end_temp
        self.cooling_rate = cooling_rate
        self.total_alns_iterations = total_alns_iterations
        
        self.current_iteration = 0
        self._current_temp = start_temp # Renamed to avoid conflict if ALNS lib also has current_temp

    @property
    def current_temp(self):
        # Exponential cooling: T_i = T_0 * (alpha ^ i)
        # Ensure temp doesn't go below end_temp
        effective_temp = self.start_temp * (self.cooling_rate ** self.current_iteration)
        return max(effective_temp, self.end_temp)

    def __call__(self, rnd_state: rnd.RandomState, best_state: State, current_state: State, candidate_state: State) -> bool:
        cand_obj = candidate_state.objective()
        curr_obj = current_state.objective()

        # Make sure objectives are calculated (they should be by ALNS framework before this)
        # The SolutionState.objective() relies on .is_feasible and .cached_objective
        # which are set by .recalculate_all_metrics_and_check_feasibility()
        # This recalc must be called by operators or ALNS framework on candidate_state
        # before this criterion is invoked. The provided ALNS lib usually does this.

        accepted = False
        if cand_obj < curr_obj:  # Minimization
            accepted = True
        else:
            # If candidate is infeasible (obj is inf), don't accept unless current is also inf
            if cand_obj == float('inf') and curr_obj != float('inf'):
                accepted = False
            elif cand_obj == float('inf') and curr_obj == float('inf'): # both infeasible
                 accepted = True # allow movement in infeasible space if current is also bad
            else: # Both are feasible, but candidate is not better
                delta = cand_obj - curr_obj
                temp = self.current_temp 
                if temp > 1e-9: # Avoid division by zero or very small temp
                    prob = math.exp(-delta / temp)
                    accepted = rnd_state.random() < prob
                else: # Temperature is effectively zero
                    accepted = False
        
        self.current_iteration += 1
        return accepted

# --- Destroy Operators ---
def random_customer_removal(current_state: SolutionState, rnd_state: rnd.RandomState, **kwargs) -> SolutionState:
    state = current_state.copy()
    
    num_customers_to_remove = kwargs.get('num_to_remove', 0)
    if num_customers_to_remove == 0: # Default if not passed
        num_customers_to_remove = math.ceil(len(state.customers_list) * DEGREE_OF_DESTRUCTION)
    
    # Identify currently served customers
    served_customers_in_routes = []
    for se_route in state.se_routes:
        for cust in se_route.customer_sequence:
            served_customers_in_routes.append(cust)
    
    if not served_customers_in_routes:
        state.recalculate_all_metrics_and_check_feasibility()
        return state # No customers to remove

    num_to_remove_actual = min(num_customers_to_remove, len(served_customers_in_routes))
    
    # Customers to be removed (actual Customer objects)
    # Ensure we don't try to remove from state.unserved_customers
    # rnd_state.choice needs a list of items to choose from, not indices if using replace=False
    
    # Create a temporary list of (customer_obj, se_route_id, index_in_route)
    candidates_for_removal = []
    for r_idx, se_route in enumerate(state.se_routes):
        for c_idx, cust in enumerate(se_route.customer_sequence):
            candidates_for_removal.append({'customer': cust, 'route_obj': se_route, 'original_idx_in_route': c_idx})

    if not candidates_for_removal:
        state.recalculate_all_metrics_and_check_feasibility()
        return state

    num_to_remove_actual = min(num_customers_to_remove, len(candidates_for_removal))
    
    # Choose indices from candidates_for_removal to pick unique customers
    chosen_indices = rnd_state.choice(len(candidates_for_removal), size=num_to_remove_actual, replace=False)
    
    customers_removed_this_op = []
    # To correctly remove from routes, it's better to collect all to remove, then iterate routes once
    
    # Map: route_obj -> list of customer_ids to remove from this route
    map_route_to_cust_ids_to_remove = {}

    for idx in chosen_indices:
        item = candidates_for_removal[idx]
        cust_obj = item['customer']
        route_obj = item['route_obj']
        
        if cust_obj.id not in [c.id for c in customers_removed_this_op]: # ensure unique customer obj if multiple entries due to error
            customers_removed_this_op.append(cust_obj)
            if route_obj not in map_route_to_cust_ids_to_remove:
                map_route_to_cust_ids_to_remove[route_obj] = []
            map_route_to_cust_ids_to_remove[route_obj].append(cust_obj.id)

    # Perform removals
    for se_route, cust_ids_to_remove in map_route_to_cust_ids_to_remove.items():
        new_sequence = [c for c in se_route.customer_sequence if c.id not in cust_ids_to_remove]
        se_route.customer_sequence = new_sequence
        # Metrics will be recalculated later, no need to call se_route.calculate_metrics() here

    for cust_obj in customers_removed_this_op:
        cust_obj.is_served = False # Mark on original customer object if it's shared
        cust_obj.assigned_se_route_id = None
        state.unserved_customers.append(cust_obj)

    # Clean up empty SE routes (optional, recalculate will handle structure)
    state.se_routes = [r for r in state.se_routes if r.customer_sequence]

    state.recalculate_all_metrics_and_check_feasibility() # Recalculate after modification
    return state


# --- Repair Operators ---
def greedy_customer_insertion(current_state: SolutionState, rnd_state: rnd.RandomState, **kwargs) -> SolutionState:
    state = current_state.copy() # Work on a copy

    if not state.unserved_customers:
        state.recalculate_all_metrics_and_check_feasibility()
        return state

    # Try to insert customers one by one. Order might matter. Shuffle for variety.
    # Make a mutable list of unserved customers to work with
    customers_to_insert_q = list(state.unserved_customers)
    rnd_state.shuffle(customers_to_insert_q)
    
    successfully_reinserted_customers = []

    for cust_to_insert in customers_to_insert_q:
        best_insertion = None # Stores (cost_increase, target_se_route, insert_idx, (optional: new_route_flag, new_route_sat_obj))
        
        # Option 1: Try inserting into existing SE routes
        for se_route in state.se_routes:
            original_len = len(se_route.customer_sequence)
            for i in range(original_len + 1):
                se_route.customer_sequence.insert(i, cust_to_insert)
                
                # Temporarily update satellite demands for this SE route's satellite
                # This is complex. A full recalc for each test is too slow.
                # SERoute.calculate_metrics checks time windows and capacity.
                # It needs the correct satellite FE arrival time.
                # Assume satellite FE times are fixed during this inner loop for speed.
                
                master_sat_obj = next((s for s in state.satellites_list if s.id == se_route.physical_satellite.id), None)
                if master_sat_obj: se_route.physical_satellite = master_sat_obj # Ensure it has up-to-date FE times
                
                if se_route.calculate_metrics(): # Check feasibility of this insertion
                    # Cost increase: new_cost - old_cost_of_route. We don't have old_cost easily.
                    # Instead, let's use total solution cost change if we had a quick way.
                    # For greedy, just find *a* feasible one. Better: find cheapest.
                    # To find cheapest, we need to calculate the cost of *this route* if cust is inserted.
                    # and compare it to the cost if not. This is tricky with the current setup.
                    # A simpler greedy: just find first valid.
                    if best_insertion is None: # or if this_insertion_is_cheaper
                         best_insertion = {'type': 'existing', 'route': se_route, 'idx': i, 'customer': cust_to_insert}
                         # To make it "best", we'd store the route cost and compare.
                         # For now, this means we'll take it and break, or undo and continue search.
                         # Let's undo and search all, then pick best.
                    
                    # Undo insertion for next test
                    se_route.customer_sequence.pop(i)
                    # se_route.calculate_metrics() # Restore route to original state metrics (important if stateful)
                else: # Insertion not feasible
                    se_route.customer_sequence.pop(i) # Undo
                    # se_route.calculate_metrics() # Restore
        
        # Option 2: Try creating a new SE route for this customer
        if best_insertion is None: # Or if creating new route is better than best_insertion to existing
            for sat_obj in state.satellites_list:
                temp_new_se_route = SERoute(id=len(state.se_routes) + 201 + len(successfully_reinserted_customers), 
                                            physical_satellite_obj=sat_obj, 
                                            vehicle_capacity=state.se_capacity)
                temp_new_se_route.customer_sequence = [cust_to_insert]
                
                master_sat_obj_for_new = next((s for s in state.satellites_list if s.id == sat_obj.id), None)
                if master_sat_obj_for_new: temp_new_se_route.physical_satellite = master_sat_obj_for_new

                if temp_new_se_route.calculate_metrics():
                    # This new route is feasible. Compare its cost to other options.
                    # For simple greedy: if no existing spot, take this new route.
                     if best_insertion is None: # Or if this new route is cheaper than inserting into existing
                        best_insertion = {'type': 'new', 'satellite': sat_obj, 'customer': cust_to_insert, 'route_obj_temp': temp_new_se_route}
                        # Again, for proper "best", compare cost. This just takes first.

        # Apply the best insertion found for cust_to_insert
        if best_insertion:
            cust_obj_inserted = best_insertion['customer']
            if best_insertion['type'] == 'existing':
                target_route = best_insertion['route']
                insert_idx = best_insertion['idx']
                target_route.customer_sequence.insert(insert_idx, cust_obj_inserted)
                # target_route.calculate_metrics() # Finalize this route's state
            elif best_insertion['type'] == 'new':
                # The route ID for SERoute needs to be unique
                new_route_obj = best_insertion['route_obj_temp'] # Already calculated
                state.se_routes.append(new_route_obj)
            
            cust_obj_inserted.is_served = True
            successfully_reinserted_customers.append(cust_obj_inserted)
        # else: customer remains unserved if no spot found

    # Update the main unserved_customers list in the state
    state.unserved_customers = [c for c in state.unserved_customers if c not in successfully_reinserted_customers]
    
    state.recalculate_all_metrics_and_check_feasibility() # Full recalculation after all insertions attempted
    return state


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Use a fixed seed for reproducibility during development
    # seed = 1234
    # random.seed(seed) # For Python's random
    # numpy_random_state = rnd.RandomState(seed) # For numpy-based random operations
    
    # For submission/actual runs, use a random seed or allow it to be time-based
    numpy_random_state = rnd.RandomState()


    input_file = "E:/LOG des/Documents/2e-vrp-pdd-main/2e-vrp-pdd-main/instance-set-2e-vrp-pdd/50 customers/A_50_5_D.csv" # Or instance from arguments
    # input_file = "test_instance_mini.csv" # A very small test instance would be good
    
    # Patch for build_initial_solution to use numpy_random_state if available
    # This requires `build_initial_solution` to accept `rnd_state_np`
    # For now, assuming `build_initial_solution` uses global `random` or is patched elsewhere.
    hub_obj, satellites_list_init, customers_list_init, fe_cap, se_cap = parse_input(input_file)
    customers_map_init = {c.id: c for c in customers_list_init}

    print(f"Parsed Data: Hub: {hub_obj}, Satellites: {len(satellites_list_init)}, Customers: {len(customers_list_init)}")
    print(f"FE Cap: {fe_cap}, SE Cap: {se_cap}")

    if not hub_obj or not customers_list_init or fe_cap < 1e-9 or se_cap < 1e-9:
        print("Error: Essential data not loaded. Exiting.")
        exit()

    # Create initial solution using the heuristic from code(4).py
    # Ensure all customer objects in routes are references to originals in customers_list_init
    initial_fe_routes, initial_se_routes = build_initial_solution(
        hub_obj, satellites_list_init, customers_list_init, fe_cap, se_cap, customers_map_init # Pass rnd_state_np if modified
    )
    
    # Check initial solution's unserved customers
    initial_unserved_customers_objs = []
    served_ids = set()
    for r in initial_se_routes:
        for c in r.customer_sequence:
            served_ids.add(c.id)
    for c_obj in customers_list_init:
        if c_obj.id not in served_ids:
            initial_unserved_customers_objs.append(c_obj)
            c_obj.is_served = False # Ensure consistency

    print(f"Initial solution: FE Routes: {len(initial_fe_routes)}, SE Routes: {len(initial_se_routes)}")
    print(f"Initial solution: Unserved customers by heuristic: {len(initial_unserved_customers_objs)}")


    initial_state = SolutionState(hub_obj, satellites_list_init, customers_list_init, customers_map_init,
                                  fe_cap, se_cap, initial_fe_routes, initial_se_routes,
                                  initial_unserved_customers_objs, numpy_random_state)
    
    print(f"Initial state objective: {initial_state.objective():.2f}, Feasible: {initial_state.is_feasible}")
    
    if not initial_state.is_feasible and not initial_unserved_customers_objs :
        print("Warning: Initial solution calculation resulted in infeasibility even with all customers supposedly served by heuristic.")
        # This might indicate issues in the initial heuristic or the SolutionState recalculation logic with the initial routes.

    if not initial_state.is_feasible and initial_unserved_customers_objs:
        print(f"Warning: Initial solution has {len(initial_unserved_customers_objs)} unserved customers, ALNS will try to fix.")
        # ALNS should ideally start from a feasible state, or one where unserved are known.
        # If initial_state.objective() is inf, SA might struggle if it can't find a feasible neighbor quickly.


    alns = ALNS(numpy_random_state)
    alns.add_destroy_operator(random_customer_removal, "RandomRemove")
    # Add more destroy operators here (e.g., worst removal, related removal)

    alns.add_repair_operator(greedy_customer_insertion, "GreedyInsert")
    # Add more repair operators here (e.g., k-regret insertion)

    num_alns_iterations = 50000 # Number of ALNS iterations - adjust based on paper/problem size

    # Weight Scheme
    # SimpleWeights(scores, num_destroy, num_repair, op_decay)
    weight_scheme = SimpleWeights(SCORES, 
                                  len(alns.destroy_operators), 
                                  len(alns.repair_operators), 
                                  REACTION_FACTOR,
                                  )
    
    # Acceptance Criterion
    acceptance_criterion = SimulatedAnnealing(SIMULATED_ANNEALING_START_TEMP,
                                              SIMULATED_ANNEALING_END_TEMP,
                                              num_alns_iterations, # Total iterations ALNS will run
                                              COOLING_RATE)

    print(f"\nStarting ALNS for {num_alns_iterations} iterations...")
    result = alns.iterate(initial_state,
                          weight_scheme,
                          acceptance_criterion,
                          iterations=num_alns_iterations,
                          # Pass kwargs for operators if needed, e.g.:
                          # num_to_remove=math.ceil(len(customers_list_init) * DEGREE_OF_DESTRUCTION) 
                          )

    best_solution_state = result.best_state
    
    final_objective = best_solution_state.objective()
    print(f"\nALNS finished. Best objective: {final_objective:.2f}")
    print(f"Best solution feasible: {best_solution_state.is_feasible}")
    print(f"Best solution unserved customers: {len(best_solution_state.unserved_customers)}")

    # Output the best solution
    # Ensure satellite FE times are correctly set in the best_solution_state for output
    # The recalculate_all_metrics should have handled this.
    best_solution_state._update_satellite_fe_times_from_fe_routes() # Ensure they are fresh for printing

    output_str = format_solution_output(best_solution_state.fe_routes,
                                        best_solution_state.se_routes,
                                        best_solution_state.customers_map,
                                        best_solution_state.hub,
                                        {}) # physical_satellites_map_ignored
    
    output_file_name = f"alns_solution_{input_file.split('/')[-1].replace('.csv', '')}.txt"
    with open(output_file_name, "w") as f_out:
        f_out.write(output_str)
    print(f"Formatted ALNS solution written to {output_file_name}")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Plot statistics to see the progress and results
    import matplotlib.pyplot as plt
    result.plot_objectives()
    plt.title("Objective Value Progression")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.show()
    result.plot_operator_counts()
    plt.title("Operator Usage Counts")
    plt.show()

    # Plot ALNS best solution routes
    plot_routes(best_solution_state.hub, best_solution_state.satellites_list, best_solution_state.customers_list,
               best_solution_state.fe_routes, best_solution_state.se_routes, title="ALNS Best Solution Routes")